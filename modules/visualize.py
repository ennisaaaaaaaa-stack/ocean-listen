"""Spectrogram visualization — 4-panel analysis chart.
Adapted from Tinggu (SeithAsync, MIT).
"""
import pathlib


def generate(audio_path, analysis_data, output_dir):
    """Generate 4-panel spectrogram: Mel, Chroma, RMS energy, Frequency bands."""
    import librosa
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    SR = 22050
    HOP = 512

    y, sr = librosa.load(str(audio_path), sr=SR)
    duration = librosa.get_duration(y=y, sr=sr)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13))

    # Mel spectrogram
    mel_s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(mel_s, ref=np.max),
                             sr=sr, x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set_title("Mel Spectrogram")

    # Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=axes[1])
    axes[1].set_title("Chromagram (Key)")

    # RMS energy
    rms = np.array(analysis_data.get("rms", []))
    times_rms = np.array(analysis_data.get("times_rms", []))
    perc_rms = np.array(analysis_data.get("perc_rms", []))
    axes[2].plot(times_rms, rms, color="#e74c3c", linewidth=0.8, label="total")
    if len(perc_rms) == len(times_rms):
        axes[2].plot(times_rms, perc_rms, color="#3498db", linewidth=0.8, label="percussive")
    axes[2].fill_between(times_rms, rms, alpha=0.3, color="#e74c3c")
    axes[2].set_title("Energy (RMS)")
    axes[2].legend()

    # Frequency bands
    BANDS = [("low", 20, 120), ("low-mid", 120, 350), ("mid", 350, 2000),
             ("high", 2000, 6000), ("air", 6000, 11025)]
    stft_mag = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr)
    band_times = librosa.times_like(stft_mag[0], sr=sr, hop_length=HOP)
    for name, low, high in BANDS:
        bins = (freqs >= low) & (freqs < high)
        band_curve = stft_mag[bins].mean(axis=0)
        # smooth
        from modules.structure import _smooth_curve
        band_smooth = _smooth_curve(band_curve, sr, np)
        axes[3].plot(band_times[:len(band_smooth)], band_smooth,
                     linewidth=0.8, label=name)
    axes[3].set_title("Frequency Bands")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()

    plt.tight_layout()
    img_path = pathlib.Path(output_dir) / f"{pathlib.Path(audio_path).stem}_analysis.png"
    plt.savefig(str(img_path), dpi=150)
    plt.close()
    return str(img_path)
