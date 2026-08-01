"""Structural analysis: BPM, key, energy, frequency bands, brightness.
Adapted from Tinggu (SeithAsync, MIT).
"""
import numpy as np

HOP = 512
SR = 22050
BAND_ENTRY_RATIO = 0.15
BAND_SUSTAIN_S = 1.0
PERC_ENTRY_RATIO = 0.25
PERC_SUSTAIN_S = 0.5
VOCAL_RATIO = 0.45
VOCAL_MIN_SEG_S = 1.5
SMOOTH_S = 0.25
BANDS = [("low", 20, 120), ("low-mid", 120, 350), ("mid", 350, 2000),
         ("high", 2000, 6000), ("air", 6000, 11025)]
SEG_COUNT = 6
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _smooth_curve(curve, sr, np_mod):
    window = max(1, round(SMOOTH_S * sr / HOP))
    return np_mod.convolve(curve, np_mod.ones(window) / window, mode="same")


def _first_sustained_entry(curve, threshold, sustain_s, sr, np_mod):
    sustain_frames = max(1, round(sustain_s * sr / HOP))
    run = 0
    for i, is_above in enumerate(curve > threshold):
        run = run + 1 if is_above else 0
        if run >= sustain_frames:
            return round(float((i - sustain_frames + 1) * HOP / sr), 1)
    return None


def _merge_short_segments(mask, min_frames):
    mask = mask.copy()
    while True:
        boundaries = [0]
        boundaries.extend(i for i in range(1, len(mask)) if mask[i] != mask[i - 1])
        boundaries.append(len(mask))
        short = next((i for i in range(len(boundaries) - 1)
                      if boundaries[i + 1] - boundaries[i] < min_frames), None)
        if short is None or len(boundaries) == 2:
            return mask
        start, end = boundaries[short], boundaries[short + 1]
        if short == 0:
            neighbor = mask[end]
        elif short == len(boundaries) - 2:
            neighbor = mask[start - 1]
        else:
            prev_len = start - boundaries[short - 1]
            next_len = boundaries[short + 2] - end
            neighbor = mask[start - 1] if prev_len >= next_len else mask[end]
        mask[start:end] = neighbor


def analyze(audio_path):
    """Run full structural analysis. Returns dict."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=SR)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    times_rms = librosa.times_like(rms, sr=sr, hop_length=HOP)

    # Key (chroma)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    dominant_key = KEYS[int(np.argmax(np.mean(chroma, axis=1)))]

    # Harmonic/percussive split
    y_harm, y_perc = librosa.effects.hpss(y)

    # Percussive entry
    perc_rms = librosa.feature.rms(y=y_perc, hop_length=HOP)[0]
    perc_smooth = _smooth_curve(perc_rms, sr, np)
    percussive_entry = _first_sustained_entry(
        perc_smooth, np.percentile(perc_rms, 90) * PERC_ENTRY_RATIO,
        PERC_SUSTAIN_S, sr, np)

    # Frequency bands
    stft_mag = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr)
    band_entries = {}
    band_curves = {}
    for name, low, high in BANDS:
        bins = (freqs >= low) & (freqs < high)
        smooth = _smooth_curve(stft_mag[bins].mean(axis=0), sr, np)
        band_curves[name] = smooth
        entry = _first_sustained_entry(smooth, smooth.max() * BAND_ENTRY_RATIO,
                                        BAND_SUSTAIN_S, sr, np)
        band_entries[name] = entry

    # Six-segment energy profile
    seg_len = len(rms) // SEG_COUNT
    segments = []
    for i in range(SEG_COUNT):
        seg = rms[i * seg_len:(i + 1) * seg_len]
        segments.append({
            "start": round(float(times_rms[i * seg_len]), 1),
            "end": round(float(times_rms[min((i + 1) * seg_len - 1, len(times_rms) - 1)]), 1),
            "avgEnergy": round(float(np.mean(seg)), 4),
            "maxEnergy": round(float(np.max(seg)), 4),
        })

    # Chroma by segment
    chroma_seg = len(chroma[0]) // SEG_COUNT
    chroma_by_seg = [KEYS[int(np.argmax(chroma[:, i * chroma_seg:(i + 1) * chroma_seg].mean(axis=1)))]
                     for i in range(SEG_COUNT)]

    # Brightness trend
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    cent_seg = len(centroid) // SEG_COUNT
    brightness = [round(float(centroid[i * cent_seg:(i + 1) * cent_seg].mean()), 0)
                  for i in range(SEG_COUNT)]
    first_b = np.mean(brightness[:2])
    last_b = np.mean(brightness[-2:])
    ratio = last_b / first_b if first_b else 1
    brightness_trend = "rising" if ratio > 1.15 else "falling" if ratio < 0.87 else "flat"

    # Vocal segments detection
    harm_stft = np.abs(librosa.stft(y_harm, hop_length=HOP))
    vocal_bins = (freqs >= 300) & (freqs < 2500)
    vocal_curve = harm_stft[vocal_bins].sum(axis=0) / (harm_stft.sum(axis=0) + np.finfo(float).eps)
    vocal_mask = _merge_short_segments(
        _smooth_curve(vocal_curve, sr, np) > VOCAL_RATIO,
        max(1, round(VOCAL_MIN_SEG_S * sr / HOP)))
    vocal_segments = []
    vocal_start = None
    for i, is_vocal in enumerate(vocal_mask):
        if is_vocal and vocal_start is None:
            vocal_start = i
        elif not is_vocal and vocal_start is not None:
            vocal_segments.append([round(float(vocal_start * HOP / sr), 1),
                                   round(float(i * HOP / sr), 1)])
            vocal_start = None
    if vocal_start is not None:
        vocal_segments.append([round(float(vocal_start * HOP / sr), 1), round(duration, 1)])

    # Build events timeline
    events = []
    for name, entry in band_entries.items():
        if entry is not None:
            events.append({"t": entry, "label": f"{name} enters"})
    if percussive_entry is not None:
        events.append({"t": percussive_entry, "label": "percussion enters"})
    for start, end in vocal_segments:
        events.extend([{"t": start, "label": "vocal in"},
                       {"t": end, "label": "vocal out"}])
    events.sort(key=lambda e: e["t"])

    return {
        "duration": round(duration, 1),
        "bpm": round(tempo),
        "key": dominant_key,
        "segments": segments,
        "chromaBySegment": chroma_by_seg,
        "brightnessBySegment": brightness,
        "brightnessTrend": brightness_trend,
        "vocalSegments": vocal_segments,
        "bandEntries": band_entries,
        "percussiveEntry": percussive_entry,
        "events": events,
        "perc_rms": perc_rms.tolist(),
        "times_rms": times_rms.tolist(),
        "rms": rms.tolist(),
    }
