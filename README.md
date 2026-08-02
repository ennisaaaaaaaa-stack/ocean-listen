# Ocean Listen / 听海

Let your AI hear audio.

Give it any audio file — music, voice, a podcast clip — it figures out what kind of sound it is, runs the right analysis pipeline, and returns structured data: MIDI notes, instrument timeline, stem separation, voice texture, lyrics.

人耳深处有三块全身最小的骨头——听小骨。它们自己不会「听」，它们的工作是把外界的振动，翻译成内耳能接收的信号。

鲸鱼在海里听。这个工具让 AI 也能在海里听。

## What it does

### Pre-classification

Before any heavy processing, Ocean Listen classifies the audio into one of four types and routes it to the right pipeline:

- **music** — rhythmic content + mixed instruments → full 6-track separation + per-stem MIDI
- **solo** — single instrument, no percussion, no voice → skip Demucs, go straight to MIDI
- **voice** — voice dominant, no instruments → f0 tracking + voice texture segmentation
- **mixed** — vocals with light backing → 2-source separation

Classification uses two signals: percussive ratio (from HPSS) as the primary discriminator, and PANNs instrument detection as secondary. When PANNs finds only voice but percussion is elevated (from consonants, breathing), it correctly overrides to voice mode.

You can also force a mode: `--mode music`, `--mode voice`, etc.

### Shallow listen (fast, ~35s per 3-min song)

- BPM, key, six-segment energy curve, brightness trend
- Frequency band entry detection (low/low-mid/mid/high/air)
- Vocal segment detection
- PANNs instrument recognition (guitar, bass, drums, piano, synth, strings, brass, organ)
- basic-pitch MIDI note extraction (pitch, velocity, duration)
- Spectrogram PNG (Mel + Chroma + RMS + Bands)

### Deep listen (slower, ~3-5 min per 3-min song)

- **Harmonic filter** — 3-stage MIDI cleanup that removes cross-stem bleed and harmonic artifacts:
  - Pitch range gate per instrument (vocals can't produce F1, bass can't produce C6)
  - Overlap dedup (when notes overlap in time, keep the strongest — fundamental beats harmonic)
  - Duration gate (removes < 50ms detection artifacts)
  - Typically removes 38% of raw notes, producing clean melody lines
- Demucs 6-track separation (vocals, drums, bass, guitar, piano, other)
- Per-track energy timeline (precise instrument entry/exit)
- Per-stem MIDI extraction — each instrument's notes, filtered and clean
- Vocal multi-part detection (pitch clustering for male/female harmony)
- Voice profile: breath ratio, airiness, loudness ratio, reverb tail
- f0 trajectory + vibrato detection
- **Voice texture profile** — two-axis fingerprint for any voice audio:
  - Axis 1: pitch IQR (texture roughness — how much pitch jumps around)
  - Axis 2: voiced_ratio (density — how continuously voiced)
  - 5-level texture label: sparse / natural / dense / dynamic / intense
  - Per-type breakdown with duration, median IQR, median density, median f0
  - f0 tracking up to 1000Hz (catches extreme vocal sounds that 500Hz cap missed)
- **Voice texture segmentation** — sliding-window analysis that breaks voice into typed segments:
  - `silence` — no voicing (breaths, pauses)
  - `sustained` — stable pitch, high voiced ratio (held notes)
  - `melodic` — pitch varies (singing)
  - `speech` — flat pitch, medium density (talking)
  - `non_vocal` — extreme pitch jumps (iqr > 150Hz, indicating non-standard vocal sounds)
  - Uses adaptive boundaries (2s window, 0.5s step) with type smoothing and segment merging
- **Voice timbre analysis** — 8-feature spectral and voice-quality fingerprint using parselmouth (Praat):
  - Spectral shape: centroid, rolloff, flatness (librosa)
  - Formants: F1, F2, F3 (vocal tract resonance)
  - Voice quality: jitter, shimmer, HNR (harmonics-to-noise ratio)
  - 4 label axes, each 4 levels:
    - Brightness: warm → neutral → bright → piercing
    - Cleanness: clean → natural → raspy → rough
    - Openness: closed → relaxed → open → wide
    - Stability: stable → slight → unstable
  - Output: `warm / clean / closed / stable` style descriptor string
  - Thresholds calibrated on 8 samples across controlled experiments (same voice: relaxed vs loud, low vs high octave)
- **Speech analysis** — patterns and dynamics of voiced content:
  - Speech rate: pseudo-syllable count per minute (from voiced burst detection)
  - Pause pattern: count, ratio, mean duration (continuous / natural pauses / frequent pauses)
  - Intonation: f0 range in semitones, variability (flat → monotone → varied → expressive)
  - Energy: RMS dynamic range in dB
  - Rhythm: voiced segment duration regularity (regular / irregular)

### Per-note dynamics contour

Every note gets a 7-field dynamic profile extracted directly from audio energy (RMS), not from MIDI velocity:

- peak_rms, mean_rms — absolute energy of this note
- attack_slope — how fast the note ramps up (sharp attack vs gradual swell)
- peak_position — where the energy peak sits within the note (0 = at the start, 1 = at the end)
- decay_rate — how quickly the note fades after peak
- relative — z-score against all notes in the piece (this note is +1.5σ = much louder than average)
- dynamic_label — pp / p / mp / mf / f / ff (based on the relative distribution)

Phrase-level dynamics (5-second sliding window):

- trend: crescendo / decrescendo / sustained
- mean_energy per window
- Detects long-range dynamic arcs (a 3-minute swell, a gradual fade-out)

This module exists because basic-pitch velocity measures model confidence, not loudness. Empirically verified: velocity-energy correlation across tested pieces ranges from -0.088 to +0.003 — essentially zero. Audio RMS is the only reliable loudness signal.

### Lyrics (optional, triple source)

- Whisper local transcription (offline, faster-whisper, multi-language)
- SenseVoice transcription (Alibaba FunASR, optimized for Chinese, with emotion detection)
- NetEase Cloud Music API (accurate timed lyrics, with duration guard)
- Local .lrc / .txt files
- Timeline alignment (lyrics + notes + instruments)

## The innovation: per-stem MIDI + harmonic filter

Neither parent project could do this alone:

- whale-listen extracts MIDI from the whole song — 1000+ notes, but no idea which instrument played which
- Tinggu separates stems and tracks instrument timing — but has no note-level data

Ocean Listen separates stems first, then runs basic-pitch on each one separately. Then it filters the results — pitch range gating removes bleed from other instruments, overlap dedup keeps only the fundamental when harmonics compete. Result: "vocals: 328 clean notes in C3-C6, bass: 87 notes in E1-E3" — every note knows which instrument it belongs to, and the notes are actually that instrument's, not a harmonic echo.

For transcription: clean melody lines that can be exported as MIDI and opened in MuseScore or any DAW.

For choreography: not just "the drums stopped here" but "the drums stopped, the bass is walking, the vocal left a 2-second breath — that's where a wave goes."

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For deep listen:

```bash
pip install -r requirements-deep.txt
```

First deep listen auto-downloads models (~330MB PANNs + ~80MB Demucs + ~900MB SenseVoice).

## Usage

```bash
# Auto-detect audio type (recommended)
python ocean.py song.mp3

# Force a specific mode
python ocean.py voice_recording.m4a --mode voice
python ocean.py piano_solo.wav --mode solo

# Deep listen (full pipeline)
python ocean.py song.mp3 --deep

# Lyrics via whisper (local)
python ocean.py recording.m4a --lyric whisper --language en

# Lyrics via SenseVoice (Chinese, with emotion detection)
python ocean.py recording.m4a --lyric sensevoice --language zh

# Lyrics via NetEase
python ocean.py song.mp3 --lyric netease --lyric-value "Shy Smith Soaked"

# Full experience
python ocean.py song.mp3 --deep --lyric whisper --language en

# Force recompute (ignore cache)
python ocean.py song.mp3 --deep --force
```

## For AI reading this

If you were handed this tool by your human, a few words:

- **The report is material, not an answer.** Timeline and numbers help you say "at 2:22 the bass entered" — but which second moved your human, only your human knows. Ask.
- **The spectrogram is worth looking at** (if you can see images): harmonic layers of vocals, vertical lines of drums, density of texture — some things numbers can't express.
- **Whisper lyrics are heard, not looked up.** They will have errors. Your human's ears are the ground truth.
- **The data tells you what happened, not what it meant.** The gap between "a 2-second silence" and "a breath before the wave" is where your human lives.
- **Voice segments are textures, not labels.** The type tells you the sound's shape — your human tells you what it means.

## Lineage

This project merges two projects:

1. **whale-listen** by migratorywhale (MIT)
   - MIDI note extraction via basic-pitch
   - Whisper lyrics transcription

2. **Tinggu 听骨** by SeithAsync (MIT)
   - Shallow/deep analysis architecture
   - PANNs instrument recognition
   - Demucs stem separation
   - Voice profile analysis
   - NetEase lyrics integration
   - Which itself incorporates **eryu** by sebastianevan200-stack (MIT)

3. **Reference project** by 盏 / Qizhan7
   - Project documentation provided as design reference during development

See NOTICES for full third-party attributions.

## License

MIT
