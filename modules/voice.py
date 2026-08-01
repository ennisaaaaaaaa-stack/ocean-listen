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
    """Extract f0 trajectory and detect vibrato.
    
    Returns: f0_values, f0_times, vibrato_info
    Uses pyin for accurate f0 tracking.
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        vocals_y, fmin=60, fmax=500, sr=sr,
        hop_length=HOP
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)

    # Vibrato detection: look for periodic f0 oscillation
    vibrato_info = {"detected": False, "average_rate": None, "average_depth_cents": None}
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 50:
        # Compute f0 derivative to find oscillation
        f0_diff = np.diff(voiced_f0)
        # Look for sign changes (oscillation)
        sign_changes = np.sum(np.abs(np.diff(np.sign(f0_diff))) > 0)
        # Rough vibrato rate estimate
        voiced_duration = len(voiced_f0) * HOP / sr
        if sign_changes > 4 and voiced_duration > 0.5:
            vib_rate = (sign_changes / 2) / voiced_duration
            if 3 < vib_rate < 12:  # vibrato is typically 4-8 Hz
                vib_depth_cents = np.std(voiced_f0) / np.mean(voiced_f0) * 1200 * 0.5
                vibrato_info = {
                    "detected": True,
                    "average_rate": round(vib_rate, 1),
                    "average_depth_cents": round(float(vib_depth_cents), 1),
                }

    return {
        "f0_values": f0.tolist(),
        "f0_times": times.tolist(),
        "vibrato": vibrato_info,
    }
