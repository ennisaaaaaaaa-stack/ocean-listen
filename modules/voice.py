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
        vocals_y, fmin=60, fmax=500, sr=sr,
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
