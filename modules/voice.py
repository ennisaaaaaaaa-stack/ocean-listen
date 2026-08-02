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
# Voice timbre analysis: spectral shape + voice quality (dimension 2)
# ---------------------------------------------------------------------------

# Calibration data (2026-08-02, 6 samples across 3 controlled experiments):
#
#   Metric      relaxed  loud   lowOct  highOct  test1  nonvocal  柳知萧  青年音
#   f0            232     320    281     479      256    454       268     117
#   centroid     2020    3066   1659    2249     1510   2332      2184    2225
#   HNR           16.2    12.1   17.8    19.7     17.0   13.0      14.8     7.8
#   jitter       0.007   0.005  0.005   0.002    0.005  0.006     0.007   0.017
#   shimmer      0.090   0.116  0.084   0.063    0.106  0.142     0.109   0.192
#   F1            627     846    532     554      510    638       747     589
#
# Key findings:
#   centroid: warm <1800 / neutral 1800-2200 / bright 2200-2800 / piercing >2800
#   HNR:      clean >15 / natural 12-15 / raspy 9-12 / rough <9
#   F1:       closed <560 / relaxed 560-700 / open 700-850 / wide >850
#   jitter:   stable <0.008 / slight 0.008-0.012 / unstable >0.012
#   shimmer:  smooth <0.10 / natural 0.10-0.14 / rough >0.14

_BRIGHTNESS_WARM = 1800
_BRIGHTNESS_BRIGHT = 2200
_BRIGHTNESS_PIERCING = 2800
_HNR_CLEAN = 15.0
_HNR_RASPY = 12.0
_HNR_ROUGH = 9.0
_F1_CLOSED = 560
_F1_OPEN = 700
_F1_WIDE = 850
_JITTER_STABLE = 0.008
_JITTER_UNSTABLE = 0.012
_SHIMMER_SMOOTH = 0.10
_SHIMMER_ROUGH = 0.14


def _brightness_label(centroid):
    if centroid < _BRIGHTNESS_WARM:
        return "warm"
    if centroid < _BRIGHTNESS_BRIGHT:
        return "neutral"
    if centroid < _BRIGHTNESS_PIERCING:
        return "bright"
    return "piercing"


def _cleanness_label(hnr):
    if hnr >= _HNR_CLEAN:
        return "clean"
    if hnr >= _HNR_RASPY:
        return "natural"
    if hnr >= _HNR_ROUGH:
        return "raspy"
    return "rough"


def _openness_label(f1):
    if f1 < _F1_CLOSED:
        return "closed"
    if f1 < _F1_OPEN:
        return "relaxed"
    if f1 < _F1_WIDE:
        return "open"
    return "wide"


def _stability_label(jitter, shimmer):
    score = 0
    if jitter > _JITTER_UNSTABLE:
        score += 2
    elif jitter > _JITTER_STABLE:
        score += 1
    if shimmer > _SHIMMER_ROUGH:
        score += 2
    elif shimmer > _SHIMMER_SMOOTH:
        score += 1
    if score <= 1:
        return "stable"
    if score <= 2:
        return "slight"
    return "unstable"


def voice_timbre_analysis(y, sr=SR):
    """Extract spectral and voice-quality features (dimension 2 of voice).

    Combines librosa spectral features with parselmouth (Praat) voice
    quality measures.  Requires praat-parselmouth.

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal (float).
    sr : int
        Sample rate.

    Returns
    -------
    dict with keys:
        spectral   -- {centroid, rolloff, flatness} in Hz (or ratio)
        formants   -- {F1, F2, F3} median values in Hz
        voice_quality -- {jitter, shimmer, hnr}
        pitch      -- {f0_median}
        labels     -- {brightness, cleanness, openness, stability}
        timbre_string -- human-readable summary
    Returns None if audio too short or parselmouth unavailable.
    """
    duration = len(y) / sr
    if duration < 0.3:
        return None

    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return None

    import librosa

    y32 = y.astype(np.float32)
    snd = parselmouth.Sound(y32, sampling_frequency=sr)

    # --- spectral shape (librosa) ---
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=y)

    centroid = float(np.median(cent))
    rolloff_val = float(np.median(rolloff))
    flatness_val = float(np.median(flatness))

    # --- formants (parselmouth / Praat) ---
    formant = snd.to_formant_burg(0.025, 5, 5000, 0.025, 50)
    f_vals = {1: [], 2: [], 3: []}
    n_samples = int(duration / 0.05)
    for i in range(n_samples):
        t = i * 0.05
        for n in (1, 2, 3):
            try:
                v = formant.get_value_at_time(n, t)
                if v and v > 0:
                    f_vals[n].append(v)
            except Exception:
                pass
    formants = {}
    for n in (1, 2, 3):
        formants[f"F{n}"] = int(np.median(f_vals[n])) if f_vals[n] else 0

    # --- pitch + jitter/shimmer (parselmouth) ---
    pitch = snd.to_pitch(0.025, 60, 1000)
    pp = call(pitch, "To PointProcess")
    try:
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter = 0.0
    try:
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = 0.0

    # --- HNR (parselmouth) ---
    harmonicity = snd.to_harmonicity_cc(0.01, 75, 0.1, 1.0)
    hnr_arr = np.array(harmonicity.values).flatten()
    hnr_vals = hnr_arr[hnr_arr > -200]
    hnr = float(np.mean(hnr_vals)) if len(hnr_vals) > 0 else 0.0

    # --- f0 median ---
    pitch_freq = pitch.selected_array["frequency"]
    voiced_f0 = pitch_freq[pitch_freq > 0]
    f0_med = float(np.median(voiced_f0)) if len(voiced_f0) else 0.0

    # --- labels ---
    brightness = _brightness_label(centroid)
    cleanness = _cleanness_label(hnr)
    openness = _openness_label(formants["F1"])
    stability = _stability_label(jitter, shimmer)

    return {
        "spectral": {
            "centroid_hz": round(centroid),
            "rolloff_hz": round(rolloff_val),
            "flatness": round(flatness_val, 4),
        },
        "formants": {
            "F1": formants["F1"],
            "F2": formants["F2"],
            "F3": formants["F3"],
        },
        "voice_quality": {
            "jitter": round(float(jitter), 4),
            "shimmer": round(float(shimmer), 4),
            "hnr_db": round(hnr, 1),
        },
        "pitch": {
            "f0_median": round(f0_med),
        },
        "labels": {
            "brightness": brightness,
            "cleanness": cleanness,
            "openness": openness,
            "stability": stability,
        },
        "timbre_string": f"{brightness} / {cleanness} / {openness} / {stability}",
    }


# ---------------------------------------------------------------------------
# Voice texture profile: multi-axis fingerprint
# ---------------------------------------------------------------------------

def voice_texture_profile(segments, timbre=None):
    """Compute a multi-dimensional voice fingerprint from segments.

    Axis 1: pitch_iqr — texture roughness (how much pitch jumps around)
    Axis 2: voiced_ratio — density (how continuously voiced the audio is)
    Axis 3 (if timbre provided): brightness — spectral centroid warm/piercing
    Axis 4 (if timbre provided): cleanness — HNR-based voice quality

    Together they form a "texture map" where different voice types
    occupy distinct regions:
        - Pure speech: low iqr (20-40), low-mid density (0.3-0.4)
        - Singing: mid iqr (40-80), high density (0.5-0.8)
        - Extreme sounds: very high iqr (150-400+), variable density

    Parameters
    ----------
    segments : list of dict
        Output from segment_voice().
    timbre : dict or None
        Output from voice_timbre_analysis().  If provided, enriches the
        profile with spectral and voice-quality data.

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

    # Include raw segment list for downstream consumers
    profile["segments"] = [
        {
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "type": s["type"],
            "median_f0": s.get("median_f0", 0),
            "pitch_iqr": s.get("pitch_iqr", 0),
            "voiced_ratio": round(s.get("voiced_ratio", 0), 2),
        }
        for s in segments
    ]

    # --- enrich with timbre data if provided ---
    if timbre:
        profile["timbre"] = {
            "spectral": timbre["spectral"],
            "formants": timbre["formants"],
            "voice_quality": timbre["voice_quality"],
            "labels": timbre["labels"],
            "timbre_string": timbre["timbre_string"],
        }

    return profile


# ---------------------------------------------------------------------------
# Speech analysis: rate, pausing, intonation, energy dynamics
# ---------------------------------------------------------------------------

def speech_analysis(f0_values, f0_times, segments, y=None, sr=SR):
    """Analyze speech patterns from f0 trajectory and segmentation.

    Computes speech-specific metrics that complement timbre and texture:

    - speech_rate: estimated syllables/min (from voiced burst counting)
    - pause_pattern: silence ratio, count, mean pause length
    - intonation: f0 range and variability within speech segments
    - energy: RMS dynamics (if audio provided)
    - rhythm: regularity of voiced segment durations

    Parameters
    ----------
    f0_values : list of float
    f0_times : list of float
    segments : list of dict
        Output from segment_voice().
    y : np.ndarray or None
        Audio signal for energy analysis.
    sr : int

    Returns dict or None if insufficient speech content.
    """
    import numpy as np

    if not segments:
        return None

    f0 = np.array(f0_values, dtype=float)
    times = np.array(f0_times)
    voiced_mask = np.isfinite(f0) & (f0 > 0)

    # Classify segments into speech vs non-speech
    speech_segs = [s for s in segments if s["type"] == "speech"]
    silence_segs = [s for s in segments if s["type"] == "silence"]
    all_voiced_segs = [s for s in segments if s["type"] in ("speech", "melodic", "sustained")]

    total_dur = sum(s["end"] - s["start"] for s in segments)
    speech_dur = sum(s["end"] - s["start"] for s in speech_segs)
    silence_dur = sum(s["end"] - s["start"] for s in silence_segs)
    voiced_dur = sum(s["end"] - s["start"] for s in all_voiced_segs)

    if total_dur < 3 or voiced_dur < 1:
        return None

    # --- speech rate (approximation) ---
    # Count voiced bursts within all voiced segments as pseudo-syllables.
    # A burst = a contiguous run of voiced frames separated by < 3 unvoiced frames.
    pseudo_syllables = 0
    for seg in all_voiced_segs:
        s_idx = np.searchsorted(times, seg["start"])
        e_idx = np.searchsorted(times, seg["end"])
        seg_voiced = voiced_mask[s_idx:e_idx]
        if len(seg_voiced) == 0:
            continue
        # Count bursts (transitions from unvoiced to voiced)
        in_burst = False
        gap = 0
        for v in seg_voiced:
            if v:
                if not in_burst:
                    pseudo_syllables += 1
                    in_burst = True
                gap = 0
            else:
                gap += 1
                if gap > 3:
                    in_burst = False

    speech_rate = round(pseudo_syllables / voiced_dur * 60) if voiced_dur > 0 else 0

    # --- pause pattern ---
    pause_count = len(silence_segs)
    pause_ratio = round(silence_dur / total_dur, 3) if total_dur > 0 else 0
    mean_pause = round(silence_dur / pause_count, 2) if pause_count > 0 else 0

    # --- intonation within voiced segments ---
    speech_f0 = []
    for seg in all_voiced_segs:
        s_idx = np.searchsorted(times, seg["start"])
        e_idx = np.searchsorted(times, seg["end"])
        seg_f0 = f0[s_idx:e_idx]
        seg_voiced = seg_f0[seg_f0 > 0]
        speech_f0.extend(seg_voiced.tolist())

    if speech_f0:
        sf0 = np.array(speech_f0)
        f0_p10 = float(np.percentile(sf0, 10))
        f0_p90 = float(np.percentile(sf0, 90))
        f0_range = round(f0_p90 - f0_p10)
        f0_median = float(np.median(sf0))
        f0_cv = round(float(np.std(sf0) / f0_median), 3) if f0_median > 0 else 0
        # Convert range to semitones for musical context
        f0_range_st = round(12 * np.log2(f0_p90 / f0_p10), 1) if f0_p10 > 0 else 0
    else:
        f0_range = f0_median = f0_cv = f0_range_st = 0

    # --- energy dynamics ---
    energy = None
    if y is not None:
        import librosa
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
        rms_times = librosa.times_like(rms, sr=sr, hop_length=HOP)
        # Get RMS in voiced segments only
        speech_rms = []
        for seg in all_voiced_segs:
            s_idx = np.searchsorted(rms_times, seg["start"])
            e_idx = np.searchsorted(rms_times, seg["end"])
            speech_rms.extend(rms[s_idx:e_idx].tolist())
        if speech_rms:
            sr_arr = np.array(speech_rms)
            energy = {
                "median_rms": round(float(np.median(sr_arr)), 5),
                "p10_rms": round(float(np.percentile(sr_arr, 10)), 5),
                "p90_rms": round(float(np.percentile(sr_arr, 90)), 5),
                "dynamic_range_db": round(20 * np.log10(
                    np.percentile(sr_arr, 90) / max(np.percentile(sr_arr, 10), 1e-8)
                ), 1),
            }

    # --- rhythm regularity ---
    voiced_durations = [s["end"] - s["start"] for s in all_voiced_segs]
    if len(voiced_durations) > 2:
        vd = np.array(voiced_durations)
        rhythm_cv = round(float(np.std(vd) / np.mean(vd)), 3) if np.mean(vd) > 0 else 0
    else:
        rhythm_cv = 0

    # --- overall labels ---
    if speech_rate > 200:
        rate_label = "fast"
    elif speech_rate > 100:
        rate_label = "moderate"
    elif speech_rate > 0:
        rate_label = "measured"
    else:
        rate_label = "none"

    if pause_ratio > 0.3:
        pause_label = "frequent_pauses"
    elif pause_ratio > 0.1:
        pause_label = "natural_pauses"
    else:
        pause_label = "continuous"

    if f0_range_st > 8:
        intonation_label = "expressive"
    elif f0_range_st > 3:
        intonation_label = "varied"
    elif f0_range_st > 0:
        intonation_label = "monotone"
    else:
        intonation_label = "flat"

    return {
        "speech_rate_syllables_per_min": speech_rate,
        "rate_label": rate_label,
        "pause": {
            "count": pause_count,
            "ratio": pause_ratio,
            "mean_duration_s": mean_pause,
            "label": pause_label,
        },
        "intonation": {
            "f0_median_hz": round(f0_median),
            "f0_range_hz": f0_range,
            "f0_range_semitones": f0_range_st,
            "f0_cv": f0_cv,
            "label": intonation_label,
        },
        "energy": energy,
        "rhythm": {
            "voiced_segment_cv": rhythm_cv,
            "label": "regular" if rhythm_cv < 0.5 else "irregular",
        },
        "content_breakdown": {
            "speech_pct": round(speech_dur / total_dur * 100) if total_dur else 0,
            "silence_pct": round(silence_dur / total_dur * 100) if total_dur else 0,
            "voiced_pct": round(voiced_dur / total_dur * 100) if total_dur else 0,
            "total_duration_s": round(total_dur, 1),
        },
    }
