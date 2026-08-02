"""Instrument recognition via PANNs Sound Event Detection.
Adapted from Tinggu (SeithAsync, MIT).
"""
import numpy as np

SED_CHUNK_S = 60
SED_SMOOTH_S = 0.5
SED_ON = 0.12
SED_OFF = 0.06
SED_MIN_SEG_S = 2.0
SED_MERGE_GAP_S = 3.0

INSTRUMENT_GROUPS = {
    "guitar": ["Guitar", "Acoustic guitar", "Electric guitar", "Strum"],
    "bass": ["Bass guitar"],
    "drums": ["Drum kit", "Drum", "Snare drum", "Bass drum", "Hi-hat", "Cymbal", "Percussion", "Tabla"],
    "piano": ["Piano", "Electric piano"],
    "synth": ["Synthesizer", "Sampler"],
    "strings": ["Violin, fiddle", "Cello", "String section", "Orchestra"],
    "brass": ["Trumpet", "Saxophone", "Flute", "Clarinet", "Trombone", "Harmonica", "Accordion"],
    "organ": ["Organ", "Electronic organ", "Hammond organ"],
    "vocals": ["Singing", "Male singing", "Female singing", "Child singing", "Choir", "Humming"],
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue"],
    "whistle": ["Whistling", "Whistle"],
}


def detect(audio_path):
    """Detect instruments using PANNs SED model."""
    import librosa
    from panns_inference import SoundEventDetection
    from panns_inference.config import labels

    audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
    duration = len(audio) / sr
    sed = SoundEventDetection(checkpoint_path=None, device="cpu")

    chunk_samples = SED_CHUNK_S * sr
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        chunks.append(sed.inference(chunk[None, :])[0])
    framewise = np.concatenate(chunks, axis=0)
    fps = len(framewise) / duration

    label_map = {label: i for i, label in enumerate(labels)}
    segments = {}
    confidence = {}

    for group, group_labels in INSTRUMENT_GROUPS.items():
        indices = [label_map[l] for l in group_labels if l in label_map]
        if not indices:
            continue
        curve = framewise[:, indices].max(axis=1)
        smooth_frames = max(1, round(SED_SMOOTH_S * fps))
        curve = np.convolve(curve, np.ones(smooth_frames) / smooth_frames, mode="same")

        raw_segs = []
        seg_start = None
        for i, prob in enumerate(curve):
            if seg_start is None and prob >= SED_ON:
                seg_start = i / fps
            elif seg_start is not None and prob < SED_OFF:
                seg_end = i / fps
                if seg_end - seg_start >= SED_MIN_SEG_S:
                    raw_segs.append([seg_start, seg_end])
                seg_start = None
        if seg_start is not None:
            if duration - seg_start >= SED_MIN_SEG_S:
                raw_segs.append([seg_start, duration])

        merged = []
        for s, e in raw_segs:
            if merged and s - merged[-1][1] < SED_MERGE_GAP_S:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        if merged:
            segments[group] = [[round(s, 1), round(e, 1)] for s, e in merged]

            # Confidence: average probability during active frames + coverage ratio
            active_mask = np.zeros(len(curve), dtype=bool)
            for s, e in merged:
                sf = int(s * fps)
                ef = min(int(e * fps), len(curve))
                active_mask[sf:ef] = True
            if np.any(active_mask):
                avg_prob = float(np.mean(curve[active_mask]))
                coverage = float(np.sum(active_mask) / len(curve))
                confidence[group] = {
                    "avg_prob": round(avg_prob, 3),
                    "coverage": round(coverage, 3),
                }

    result = dict(segments)
    if confidence:
        result["_confidence"] = confidence
    return result
