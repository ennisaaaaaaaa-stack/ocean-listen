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
- **Voice texture segmentation** — sliding-window analysis that breaks voice into typed segments:
  - `silence` — no voicing (breaths, pauses)
  - `sustained` — stable pitch, high voiced ratio (held notes)
  - `melodic` — pitch varies (singing)
  - `speech` — flat pitch, medium density (talking)
  - `non_vocal` — extreme pitch jumps (iqr > 150Hz, indicating non-standard vocal sounds)
  - Uses adaptive boundaries (2s window, 0.5s step) with type smoothing and segment merging

### Lyrics (optional, dual source)

- Whisper local transcription (offline, faster-whisper, multi-language)
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

First deep listen auto-downloads models (~330MB PANNs + ~80MB Demucs).

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

See NOTICES for full third-party attributions.

## License

MIT
