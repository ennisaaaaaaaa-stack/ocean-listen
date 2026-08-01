"""Demucs stem separation and per-track timeline analysis.
Adapted from Tinggu (SeithAsync, MIT).
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

TRACKS = ("vocals", "drums", "bass", "guitar", "piano", "other")
SR = 22050
HOP = 512
STEM_SMOOTH_S = 0.3
STEM_MIN_SEG_S = 1.5
STEM_MERGE_GAP_S = 2.5
STEM_SIGNAL_FRACTION = 0.08  # fraction of (peak - noise_floor) above noise_floor


def split(audio_path, destination):
    """Run Demucs 6-source separation."""
    if all((destination / f"{t}.mp3").exists() for t in TRACKS):
        return
    destination.mkdir(parents=True, exist_ok=True)
    temp = pathlib.Path(tempfile.mkdtemp(prefix="stems-", dir=destination.parent))
    try:
        subprocess.run([
            sys.executable, "-m", "demucs", "-n", "htdemucs_6s", "--mp3",
            "--mp3-bitrate", "128", "-j", "2", "-o", str(temp), str(audio_path),
        ], check=True, timeout=1800)
        source = temp / "htdemucs_6s" / pathlib.Path(audio_path).stem
        for track in TRACKS:
            src = source / f"{track}.mp3"
            if not src.exists():
                raise RuntimeError(f"Demucs output missing {track}.mp3")
            dst = destination / f"{track}.mp3"
            dst.unlink(missing_ok=True)
            shutil.move(str(src), str(dst))
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def _smooth_rms(y, librosa, np_mod):
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    width = max(1, round(STEM_SMOOTH_S * SR / HOP))
    return np_mod.convolve(rms, np_mod.ones(width) / width, mode="same")


def _active_segments(rms, np_mod):
    if not len(rms) or not np_mod.any(rms):
        return []
    # Adaptive threshold: noise_floor + fraction of dynamic range
    noise_floor = float(np_mod.percentile(rms, 20))
    peak = float(np_mod.percentile(rms, 98))
    threshold = noise_floor + STEM_SIGNAL_FRACTION * (peak - noise_floor)
    active = rms > threshold
    edges = np_mod.diff(np_mod.pad(active.astype(np.int8), (1, 1)))
    starts = np_mod.flatnonzero(edges == 1) * HOP / SR
    ends = np_mod.flatnonzero(edges == -1) * HOP / SR
    segs = [[float(s), float(e)] for s, e in zip(starts, ends) if e - s >= STEM_MIN_SEG_S]
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < STEM_MERGE_GAP_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [[round(s, 1), round(e, 1)] for s, e in merged]


def build_timeline(stems_dir):
    """Build per-track active/inactive timeline from separated stems."""
    import librosa
    timeline = {}
    vocals_y = None
    vocals_rms = None

    for track in TRACKS:
        track_path = stems_dir / f"{track}.mp3"
        if not track_path.exists():
            timeline[track] = []
            continue
        y, _ = librosa.load(track_path, sr=SR, mono=True)
        rms = _smooth_rms(y, librosa, np)
        timeline[track] = _active_segments(rms, np)
        if track == "vocals":
            vocals_y, vocals_rms = y, rms

    return timeline, vocals_y, vocals_rms
