"""Voice profile: breath, air, loudness, reverb tail, vibrato, attack.
Adapted from Tinggu (SeithAsync, MIT). Extended for Ocean Listen.
"""
import numpy as np

SR = 22050
HOP = 512
VOICE_WIN_S = 25
VOICE_ACTIVE_RATIO = 0.05
AIR_HZ = 5000
TAIL_MAX_S = 8


def _mmss(seconds):
    seconds = round(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _window_metrics(y, start, librosa, np_mod):
    """Measure breath ratio, air ratio, rms in a window."""
    beginning = round(start * SR)
    segment = y[beginning:beginning + round(VOICE_WIN_S * SR)]
    _, percussive = librosa.effects.hpss(segment)
    total = float(np_mod.sum(segment ** 2))
    breath = float(np_mod.sum(percussive ** 2) / total) if total else 0.0
    spectrum = np_mod.abs(librosa.stft(segment))
    frequencies = librosa.fft_frequencies(sr=SR)
    spec_sum = float(np_mod.sum(spectrum))
    air = float(np_mod.sum(spectrum[frequencies > AIR_HZ]) / spec_sum) if spec_sum else 0.0
    rms = float(np_mod.mean(librosa.feature.rms(y=segment)[0]))
    return {"start": round(start, 1), "breathNoiseRatio": round(breath, 4),
            "airRatio": round(air, 4), "rms": rms}


def _tail_reverb(y, vocal_segments, librosa, np_mod):
    if not vocal_segments:
        return None
    start = round(vocal_segments[-1][1] * SR)
    tail = y[start:start + round(TAIL_MAX_S * SR)]
    if not len(tail):
        return None
    rms = librosa.feature.rms(y=tail, hop_length=HOP)[0]
    if not len(rms) or not np_mod.any(rms):
        return None
    peak = int(np_mod.argmax(rms))
    below = np_mod.flatnonzero(rms[peak:] <= rms[peak] * 0.1)
    if not len(below):
        return None
    return round(float(below[0] * HOP / SR), 2)


def profile(vocals_y, vocals_rms, vocal_segments, librosa, np_mod):
    """Full voice profile analysis."""
    threshold = np_mod.percentile(vocals_rms, 98) * VOICE_ACTIVE_RATIO
    active = vocals_rms > threshold
    if np_mod.count_nonzero(active) * HOP / SR < 10:
        return None

    active_idx = np_mod.flatnonzero(active)
    first = active_idx[0] * HOP / SR
    last = active_idx[-1] * HOP / SR
    win_frames = round(VOICE_WIN_S * SR / HOP)
    candidates = []
    for start in np_mod.arange(np_mod.ceil(first), np_mod.floor(last) + 0.001, 1.0):
        frame = round(start * SR / HOP)
        win_rms = vocals_rms[frame:frame + win_frames]
        win_active = active[frame:frame + win_frames]
        if np_mod.count_nonzero(win_active) < win_frames / 2:
            continue
        candidates.append((float(np_mod.mean(win_rms[win_active])), float(start)))
    if not candidates:
        return None

    soft = _window_metrics(vocals_y, min(candidates)[1], librosa, np_mod)
    burst = _window_metrics(vocals_y, max(candidates)[1], librosa, np_mod)
    loudness = burst["rms"] / soft["rms"] if soft["rms"] else None

    return {
        "softWindow": soft,
        "burstWindow": burst,
        "loudnessRatio": round(loudness, 1) if loudness else None,
        "tailReverb": _tail_reverb(vocals_y, vocal_segments, librosa, np_mod),
    }


def f0_analysis(vocals_y, sr=SR):
    """Extract f0 trajectory and detect vibrato via autocorrelation.

    Returns: f0_values, f0_times, vibrato_info
    Uses pyin for accurate f0 tracking.
    Vibrato detected by autocorrelating detrended f0 contour segments.
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        vocals_y, fmin=60, fmax=1000, sr=sr,
        hop_length=HOP
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)

    # Vibrato detection via autocorrelation on sustained voiced segments
    vibrato_info = {"detected": False, "average_rate": None, "average_depth_cents": None}

    # Find continuous voiced segments (gap tolerance = 3 frames)
    voiced_idx = np.flatnonzero(voiced_flag)
    if len(voiced_idx) < 50:
        return {
            "f0_values": f0.tolist(),
            "f0_times": times.tolist(),
            "vibrato": vibrato_info,
        }

    # Split into sustained segments at gaps > 3 frames
    gaps = np.diff(voiced_idx)
    breaks = np.flatnonzero(gaps > 3)
    seg_starts = np.concatenate([[voiced_idx[0]], voiced_idx[breaks + 1]])
    seg_ends = np.concatenate([voiced_idx[breaks], [voiced_idx[-1]]])

    min_seg_frames = int(0.3 * sr / HOP)  # min 0.3s sustained for vibrato
    vib_rates = []
    vib_depths = []

    for s, e in zip(seg_starts, seg_ends):
        seg_len = e - s + 1
        if seg_len < min_seg_frames:
            continue

        seg_f0 = f0[s:e + 1].copy()
        # Detrend: subtract moving average (window ~0.5s) to isolate oscillation
        ma_win = max(int(0.5 * sr / HOP), 5)
        if seg_len < ma_win * 2:
            # Short segment: just subtract mean
            seg_f0_detrended = seg_f0 - np.mean(seg_f0)
        else:
            # Moving average detrend
            kernel = np.ones(ma_win) / ma_win
            trend = np.convolve(seg_f0, kernel, mode='same')
            seg_f0_detrended = seg_f0 - trend

        # Autocorrelation
        seg_centered = seg_f0_detrended - np.mean(seg_f0_detrended)
        autocorr = np.correlate(seg_centered, seg_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # right half
        autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr

        # Look for first peak in 3-10 Hz range
        frame_rate = sr / HOP  # frames per second
        min_lag = int(frame_rate / 10)  # 10 Hz
        max_lag = int(frame_rate / 3)   # 3 Hz
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        if min_lag >= max_lag:
            continue

        search = autocorr[min_lag:max_lag + 1]
        if len(search) < 2:
            continue

        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(search) - 1):
            if search[i] > search[i - 1] and search[i] >= search[i + 1]:
                peaks.append((search[i], i + min_lag))

        if not peaks:
            continue

        # Best peak = highest autocorrelation
        best_corr, best_lag = max(peaks, key=lambda p: p[0])

        if best_corr < 0.15:  # too weak, not a real oscillation
            continue

        vib_rate = frame_rate / best_lag
        if not (3 <= vib_rate <= 10):
            continue

        # Depth: peak-to-peak amplitude of the oscillation in cents
        seg_mean_f0 = np.mean(seg_f0[seg_f0 > 0])
        if seg_mean_f0 <= 0:
            continue
        # Amplitude = std of detrended f0 (half of peak-to-peak for sinusoid)
        vib_amp_hz = np.std(seg_f0_detrended) * np.sqrt(2)  # peak deviation for sinusoid
        vib_depth_cents = 1200 * np.log2(1 + vib_amp_hz / seg_mean_f0) if seg_mean_f0 > 0 else 0

        vib_rates.append(vib_rate)
        vib_depths.append(vib_depth_cents)

    if vib_rates:
        vibrato_info = {
            "detected": True,
            "average_rate": round(float(np.median(vib_rates)), 1),
            "average_depth_cents": round(float(np.median(vib_depths)), 1),
            "segment_count": len(vib_rates),
        }

    return {
        "f0_values": f0.tolist(),
        "f0_times": times.tolist(),
        "vibrato": vibrato_info,
    }


# ---------------------------------------------------------------------------
# Voice segmentation: time-axis classification of voice texture
# ---------------------------------------------------------------------------

# Sliding window config
_SEG_WIN_S = 2.0       # feature window
_SEG_STEP_S = 0.5      # step between windows
_SEG_SMOOTH = 3        # median filter kernel for type smoothing
_SEG_MERGE_TOL = 2     # max consecutive different windows to bridge when merging
_SEG_MIN_DUR_S = 0.8   # segments shorter than this get absorbed

# Classification thresholds (calibrated on Whoregasm, 2026-08-02)
_SILENCE_VR = 0.15     # voiced_ratio below this = silence
_NONVOCAL_IQR = 150    # pitch IQR above this = non-vocal (BeyondWords threshold)
_SUSTAINED_VR = 0.50   # voiced_ratio above this + low cv = sustained
_SUSTAINED_CV = 0.08   # cv below this = stable pitch
_MELODIC_CV = 0.10     # cv above this = pitch varies enough to be melodic
_MELODIC_IQR = 40      # or IQR above this = melodic


def _classify_window(med_f0, iqr, voiced_ratio, cv):
    """Classify a single feature window."""
    if voiced_ratio < _SILENCE_VR:
        return "silence"
    if iqr > _NONVOCAL_IQR:
        return "non_vocal"
    if voiced_ratio > _SUSTAINED_VR and cv < _SUSTAINED_CV:
        return "sustained"
    if cv > _MELODIC_CV or iqr > _MELODIC_IQR:
        return "melodic"
    return "speech"


def segment_voice(f0_values, f0_times, sr=SR, hop=HOP):
    """Segment voice into typed regions along the time axis.

    Uses a sliding window over the f0 trajectory.  For each window it
    computes four features (median f0, pitch IQR, voiced ratio, pitch CV)
    then classifies the window.  Adjacent same-class windows are merged
    with a tolerance for brief anomalies.

    Types:
        silence    -- no voice
        sustained  -- dense voiced, stable pitch (held notes, rap)
        melodic    -- pitch varies significantly (singing)
        speech     -- sparse voiced, natural intonation
        non_vocal  -- extreme pitch range (moans, slides, gasps)

    Returns a list of segment dicts:
        {type, start, end, duration, median_f0, pitch_iqr, voiced_ratio, pitch_cv}
    """
    f0 = np.array(f0_values, dtype=float)
    times = np.array(f0_times)
    voiced_mask = np.isfinite(f0) & (f0 > 0)

    win_frames = int(_SEG_WIN_S * sr / hop)
    step_frames = int(_SEG_STEP_S * sr / hop)

    # --- feature extraction per window ---
    feats = []  # (time, type, median_f0, iqr, voiced_ratio, cv)
    for i in range(0, max(len(f0) - win_frames, 0), step_frames):
        win = f0[i:i + win_frames]
        mask = voiced_mask[i:i + win_frames]
        voiced = win[mask]
        vr = len(voiced) / len(win) if len(win) else 0
        t = float(times[i]) if i < len(times) else 0.0

        if len(voiced) < 3:
            feats.append([t, "silence", 0, 0, vr, 0])
            continue

        med = float(np.median(voiced))
        p25, p75 = np.percentile(voiced, [25, 75])
        iqr = float(p75 - p25)
        std = float(np.std(voiced))
        cv = std / med if med > 0 else 0
        tp = _classify_window(med, iqr, vr, cv)
        feats.append([t, tp, round(med), round(iqr), round(vr, 2), round(cv, 3)])

    if not feats:
        return []

    # --- type smoothing: remove single-window anomalies ---
    types = [f[1] for f in feats]
    k = _SEG_SMOOTH // 2
    for i in range(len(types)):
        lo = max(0, i - k)
        hi = min(len(types), i + k + 1)
        neighbours = types[lo:i] + types[i + 1:hi]
        if neighbours and all(n == neighbours[0] for n in neighbours) and neighbours[0] != types[i]:
            types[i] = neighbours[0]

    # --- merge consecutive same-type windows (with tolerance) ---
    segments = []
    i = 0
    while i < len(feats):
        tp = types[i]
        j = i
        skip = 0
        last_good = i
        while j < len(feats):
            if types[j] == tp:
                last_good = j
                skip = 0
            else:
                skip += 1
                if skip > _SEG_MERGE_TOL:
                    break
            j += 1

        start_t = feats[i][0]
        if last_good + 1 < len(feats):
            end_t = feats[last_good + 1][0]
        else:
            end_t = start_t + _SEG_WIN_S

        # aggregate features of matching windows only
        matched = [feats[k] for k in range(i, last_good + 1) if types[k] == tp]
        f0_vals = [f[2] for f in matched if f[2] > 0]
        med_f0 = int(np.median(f0_vals)) if f0_vals else 0
        avg_iqr = int(np.median([f[3] for f in matched])) if matched else 0
        avg_vr = round(float(np.median([f[4] for f in matched])), 2) if matched else 0
        avg_cv = round(float(np.median([f[5] for f in matched])), 3) if matched else 0

        segments.append({
            "type": tp,
            "start": round(start_t, 1),
            "end": round(end_t, 1),
            "duration": round(end_t - start_t, 1),
            "median_f0": med_f0,
            "pitch_iqr": avg_iqr,
            "voiced_ratio": avg_vr,
            "pitch_cv": avg_cv,
        })
        i = last_good + 1

    # --- absorb sub-minimum segments into neighbours ---
    merged = []
    for seg in segments:
        if seg["duration"] < _SEG_MIN_DUR_S and merged:
            prev = merged[-1]
            prev["end"] = seg["end"]
            prev["duration"] = round(prev["end"] - prev["start"], 1)
        else:
            if merged and seg["type"] == merged[-1]["type"]:
                # same type as previous after merge — combine
                prev = merged[-1]
                prev["end"] = seg["end"]
                prev["duration"] = round(prev["end"] - prev["start"], 1)
            else:
                merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# Voice texture profile: two-axis fingerprint
# ---------------------------------------------------------------------------

def voice_texture_profile(segments):
    """Compute a two-dimensional voice fingerprint from segments.

    Axis 1: pitch_iqr — texture roughness (how much pitch jumps around)
    Axis 2: voiced_ratio — density (how continuously voiced the audio is)

    Together they form a "texture map" where different voice types
    occupy distinct regions:
        - Pure speech: low iqr (20-40), low-mid density (0.3-0.4)
        - Singing: mid iqr (40-80), high density (0.5-0.8)
        - Extreme sounds: very high iqr (150-400+), variable density

    Returns a summary dict with per-type and overall statistics.
    """
    if not segments:
        return None

    # Overall stats across all segments (weighted by duration)
    total_dur = sum(s["end"] - s["start"] for s in segments)

    all_iqr = []
    all_vr = []
    for s in segments:
        dur = s["end"] - s["start"]
        weight = max(1, int(dur * 10))  # weight by 100ms units
        all_iqr.extend([s.get("pitch_iqr", 0)] * weight)
        all_vr.extend([s.get("voiced_ratio", 0)] * weight)

    import numpy as np

    profile = {
        "overall": {
            "median_iqr": int(np.median(all_iqr)),
            "p90_iqr": int(np.percentile(all_iqr, 90)),
            "median_voiced_ratio": round(float(np.median(all_vr)), 2),
            "mean_voiced_ratio": round(float(np.mean(all_vr)), 2),
            "duration_s": round(total_dur, 1),
        },
        "by_type": {},
    }

    # Per-type breakdown
    type_names = ["melodic", "speech", "sustained", "non_vocal", "silence"]
    for t in type_names:
        tsegs = [s for s in segments if s["type"] == t]
        if not tsegs:
            continue
        t_dur = sum(s["end"] - s["start"] for s in tsegs)
        t_iqr = [s.get("pitch_iqr", 0) for s in tsegs]
        t_vr = [s.get("voiced_ratio", 0) for s in tsegs]
        t_f0 = [s.get("median_f0", 0) for s in tsegs if s.get("median_f0", 0) > 0]

        profile["by_type"][t] = {
            "segments": len(tsegs),
            "duration_s": round(t_dur, 1),
            "duration_pct": round(t_dur / total_dur * 100) if total_dur else 0,
            "median_iqr": int(np.median(t_iqr)),
            "median_voiced_ratio": round(float(np.median(t_vr)), 2),
            "median_f0": int(np.median(t_f0)) if t_f0 else 0,
        }

    # Texture classification (simple quadrant)
    med_iqr = profile["overall"]["median_iqr"]
    med_vr = profile["overall"]["median_voiced_ratio"]

    if med_iqr > 100:
        texture_label = "intense"
    elif med_iqr > 50:
        texture_label = "dynamic"
    elif med_vr > 0.5:
        texture_label = "dense"
    elif med_vr > 0.25:
        texture_label = "natural"
    else:
        texture_label = "sparse"

    profile["texture_label"] = texture_label

    return profile
