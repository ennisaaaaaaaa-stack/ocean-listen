# Ocean Listen / 听海

Let your AI hear music.

Give it a local audio file, it returns structured data — MIDI notes, instrument timeline, stem separation, voice profile, lyrics — so any LLM can hear the shape of a song.

人耳深处有三块全身最小的骨头——听小骨。它们自己不会「听」，它们的工作是把外界的振动，翻译成内耳能接收的信号。

鲸鱼在海里听。这个工具让 AI 也能在海里听。

## What it does

Three layers of listening:

**Shallow listen** (fast, ~35s per 3-min song)
- BPM, key, six-segment energy curve, brightness trend
- Frequency band entry detection (low/low-mid/mid/high/air)
- Vocal segment detection
- PANNs instrument recognition (guitar, bass, drums, piano, synth, strings, brass, organ)
- basic-pitch MIDI note extraction (pitch, velocity, duration)
- Spectrogram PNG (Mel + Chroma + RMS + Bands)

**Deep listen** (slower, ~3-5 min per 3-min song)
- Demucs 6-track separation (vocals, drums, bass, guitar, piano, other)
- Per-track energy timeline (precise instrument entry/exit)
- Per-stem MIDI extraction — each instrument's notes, not just the whole song
- Vocal multi-part detection (pitch clustering for male/female harmony)
- Voice profile: breath ratio, airiness, loudness ratio, reverb tail
- f0 trajectory + vibrato detection

**Lyrics** (optional, dual source)
- Whisper local transcription (offline, faster-whisper, multi-language)
- NetEase Cloud Music API (accurate timed lyrics, with duration guard)
- Local .lrc / .txt files
- Timeline alignment (lyrics + notes + instruments)

## The innovation: per-stem MIDI

Neither parent project could do this alone:

- whale-listen extracts MIDI from the whole song — 1000+ notes, but no idea which instrument played which
- Tinggu separates stems and tracks instrument timing — but has no note-level data

Ocean Listen separates stems first, then runs basic-pitch on each one separately. Result: "drums: 200 notes, bass: 300 notes, vocals: 400 notes" — every note knows which instrument it belongs to.

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
# Shallow listen
python ocean.py song.mp3

# Deep listen
python ocean.py song.mp3 --deep

# Lyrics via whisper (local)
python ocean.py song.mp3 --lyric whisper --language en

# Lyrics via NetEase
python ocean.py song.mp3 --lyric netease --lyric-value "Shy Smith Soaked"

# Full experience
python ocean.py song.mp3 --deep --lyric whisper --language en

# Force recompute
python ocean.py song.mp3 --deep --force
```

## For AI reading this

If you were handed this tool by your human, a few words:

- **The report is material, not an answer.** Timeline and numbers help you say "at 2:22 the bass entered" — but which second moved your human, only your human knows. Ask.
- **The spectrogram is worth looking at** (if you can see images): harmonic layers of vocals, vertical lines of drums, density of texture — some things numbers can't express.
- **Whisper lyrics are heard, not looked up.** They will have errors. Your human's ears are the ground truth.
- **The data tells you what happened, not what it meant.** The gap between "a 2-second silence" and "a breath before the wave" is where your human lives.

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
