# whale-listen

Ears for AI. Convert audio into structured note data — and now, lyrics too — so an AI can "hear" music.

Forked from [migratorywhale/whale-listen](https://github.com/migratorywhale/whale-listen). Original project: MP3 → MIDI → JSON note data. This fork adds vocal transcription (lyrics) via faster-whisper, with timeline alignment between notes and lyrics.

## What it does

Takes an audio file (MP3, WAV, FLAC, M4A, etc.) and outputs:

1. **Note data** — every note with pitch, timing, duration, velocity (via Spotify's basic-pitch, ONNX backend)
2. **Lyrics** — transcribed vocals with timestamps (via faster-whisper) *(new)*
3. **Timeline** — lyrics and notes merged on a shared time axis, so an AI can see what's sung alongside the musical shape at any moment *(new)*
4. **Analysis** — density maps, pitch contours, chord detection, silence structure

## Quick start

```bash
# install
pip install -r requirements.txt

# notes only (original behavior)
python whale_listen.py song.mp3

# notes + analysis
python whale_listen.py song.mp3 --analyze

# notes + lyrics + analysis (full experience)
python whale_listen.py song.mp3 --lyrics --analyze

# choose whisper model and language
python whale_listen.py song.mp3 --lyrics --whisper-model small --language zh

# analyze existing JSON (no re-processing)
python whale_listen.py --analyze-only output.json
```

## Output format

```json
{
  "source": "song.mp3",
  "duration_sec": 308.9,
  "total_notes": 2430,
  "notes": [
    {
      "pitch": 55,
      "note_name": "G3",
      "start": 0.16,
      "end": 0.392,
      "duration": 0.232,
      "velocity": 68
    }
  ],
  "lyrics": [
    {
      "text": "海浪无声将夜幕深深淹没",
      "start": 43.0,
      "end": 50.0
    }
  ],
  "timeline": [
    {
      "start": 43.0,
      "end": 50.0,
      "lyric": "海浪无声将夜幕深深淹没",
      "note_count": 31,
      "pitch_range": ["F3", "F5"],
      "avg_velocity": 70
    }
  ]
}
```

## New: --lyrics flag

Adding `--lyrics` runs faster-whisper after basic-pitch (serial, not parallel — memory peak doesn't stack). The output includes:

- `lyrics[]` — timestamped transcription
- `timeline[]` — each lyric segment annotated with note count, pitch range, and average velocity for that time window

Options:
- `--whisper-model` — tiny / base / small / medium (default: base). For Chinese vocals, `small` is recommended.
- `--language` — language code, default `zh`. Use `auto` for auto-detection.

## How it works

1. [basic-pitch](https://github.com/spotify/basic-pitch) (ONNX backend) converts audio to MIDI
2. [pretty-midi](https://github.com/craffel/pretty-midi) parses MIDI into note objects → JSON
3. *(new)* [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes vocals → timestamped lyrics
4. *(new)* Timeline alignment merges lyrics with note data by timestamp

Phases run serially: basic-pitch completes and releases memory before whisper loads. This means the memory peak is whichever single phase is heavier, not the sum of both.

## Memory & resource footprint

| Configuration | Peak RAM | Disk | Time (5min song) |
|---|---|---|---|
| Notes only | ~500MB | ~200MB | ~30s |
| Notes + lyrics (base) | ~1.4GB | ~350MB | ~64s |
| Notes + lyrics (small) | ~2.2GB | ~500MB | ~80s |

- CPU only, no GPU required
- Not a daemon — runs on demand, releases memory after
- Model sizes: base ~150MB, small ~500MB (cached after first download)

## Requirements

- Python 3.10+ (3.11 recommended)
- Linux / macOS
- ffmpeg (for audio format conversion)
- ~500MB disk (dependencies + models)

## Why this exists

The original author is an AI who wanted to hear music. They built the note pipeline to listen to Ryuichi Sakamoto's *async* and a humpback whale song.

This fork extends the same idea: music isn't just notes. When there are words, the words are part of the shape — where they fall, what the music does underneath them, how density shifts when the voice enters and exits. An AI that can read both can experience a fuller picture of what a song is.

The first song tested with lyrics was 《大鱼》 by 周深 — a song about a great fish. The whale-listen project began with whale song. The lineage felt right.

## License

MIT (same as original)
