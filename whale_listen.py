"""
whale-listen: ears for AI
MP3 → MIDI → JSON, so an AI can "hear" music as structured data.

With --lyrics: also transcribes vocals via faster-whisper, aligning
lyrics with note data on the timeline so an AI can read what is sung
alongside the musical shape.

Usage:
    python whale_listen.py song.mp3                    # → song.json
    python whale_listen.py song.mp3 -o output.json     # custom output path
    python whale_listen.py song.mp3 --analyze          # also print analysis
    python whale_listen.py song.mp3 --lyrics           # also transcribe vocals
    python whale_listen.py song.mp3 --lyrics --analyze # full experience
"""
import argparse
import json
import pathlib
import sys

# basic-pitch uses a legacy scipy path — patch before import
import scipy.signal
from scipy.signal.windows import gaussian as _gaussian
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = _gaussian

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi


def find_model():
    """Find the ONNX model bundled with basic-pitch."""
    onnx = pathlib.Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
    if onnx.exists():
        return onnx
    # fallback: try the default saved model path
    return ICASSP_2022_MODEL_PATH


def transcribe(audio_path: str, model_size: str = "base", language: str = "zh"):
    """Transcribe vocals using faster-whisper. Returns list of lyric segments."""
    from faster_whisper import WhisperModel

    print(f"Loading whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing vocals in {pathlib.Path(audio_path).name}...")
    lang = None if language == "auto" else language
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=lang,
    )

    lyrics = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            lyrics.append({
                "text": text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
            })

    print(f"Lyrics: {len(lyrics)} segments, detected language: {info.language}")
    return lyrics


def convert(audio_path: str, output_json: str = None, output_midi: str = None,
            do_lyrics: bool = False, whisper_model: str = "base",
            language: str = "zh"):
    """Convert audio to MIDI and JSON note data. Optionally transcribe lyrics."""

    # ── Phase 1: Note detection (basic-pitch) ──
    audio = pathlib.Path(audio_path)
    if not audio.exists():
        print(f"File not found: {audio}", file=sys.stderr)
        sys.exit(1)

    model = find_model()
    print(f"Listening to {audio.name}...")

    model_output, midi_data, note_events = predict(str(audio), model)

    # write MIDI
    midi_path = output_midi or audio.with_suffix(".mid")
    midi_data.write(str(midi_path))
    print(f"MIDI → {midi_path} ({len(note_events)} note events)")

    # build JSON
    notes = []
    for inst in midi_data.instruments:
        for note in inst.notes:
            notes.append({
                "pitch": int(note.pitch),
                "note_name": pretty_midi.note_number_to_name(note.pitch),
                "start": round(float(note.start), 3),
                "end": round(float(note.end), 3),
                "duration": round(float(note.end - note.start), 3),
                "velocity": int(note.velocity),
            })

    notes.sort(key=lambda n: n["start"])

    total_duration = round(notes[-1]["end"], 1) if notes else 0

    result = {
        "source": audio.name,
        "duration_sec": total_duration,
        "total_notes": len(notes),
        "notes": notes,
    }

    # ── Phase 2: Lyrics transcription (serial — basic-pitch already released) ──
    if do_lyrics:
        lyrics = transcribe(str(audio), model_size=whisper_model,
                            language=language)
        result["lyrics"] = lyrics

        # build aligned timeline: merge notes and lyrics by time
        timeline = build_timeline(notes, lyrics)
        result["timeline"] = timeline

    json_path = output_json or audio.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    summary = f"JSON → {json_path} ({len(notes)} notes"
    if do_lyrics:
        summary += f", {len(lyrics)} lyric segments"
    summary += ")"
    print(summary)
    return result


def build_timeline(notes: list, lyrics: list):
    """
    Merge note density and lyrics into a unified timeline.
    Each segment covers one lyric line, annotated with the notes active during it.
    """
    if not lyrics:
        return []

    timeline = []
    for lyric in lyrics:
        seg_start = lyric["start"]
        seg_end = lyric["end"]

        # notes active during this lyric segment
        active = [n for n in notes if n["start"] < seg_end and n["end"] > seg_start]

        entry = {
            "start": seg_start,
            "end": seg_end,
            "lyric": lyric["text"],
        }

        if active:
            pitches = [n["pitch"] for n in active]
            entry["note_count"] = len(active)
            entry["pitch_range"] = [
                pretty_midi.note_number_to_name(min(pitches)),
                pretty_midi.note_number_to_name(max(pitches)),
            ]
            entry["avg_velocity"] = round(
                sum(n["velocity"] for n in active) / len(active)
            )

        timeline.append(entry)

    return timeline


def analyze(data: dict):
    """Print a structural analysis of note data (and lyrics if present)."""
    notes = data["notes"]
    lyrics = data.get("lyrics", [])

    if not notes and not lyrics:
        print("Nothing found.")
        return

    total_dur = data.get("duration_sec",
                         notes[-1]["end"] if notes else 0)

    print(f"\n{'=' * 60}")
    print(f"  {data.get('source', 'unknown')}")
    if notes:
        print(f"  {len(notes)} notes over {total_dur}s ({total_dur/60:.1f} min)")
    if lyrics:
        print(f"  {len(lyrics)} lyric segments")
    print(f"{'=' * 60}")

    if notes:
        pitches = [n["pitch"] for n in notes]
        velocities = [n["velocity"] for n in notes]
        durations = [n["duration"] for n in notes]

        # pitch range
        low = min(notes, key=lambda n: n["pitch"])
        high = max(notes, key=lambda n: n["pitch"])
        print(f"\nPitch range: {low['note_name']} ({low['pitch']}) — "
              f"{high['note_name']} ({high['pitch']})")

        # most common notes
        from collections import Counter
        pitch_counts = Counter(n["note_name"] for n in notes)
        print("Most common: " + ", ".join(
            f"{name} ({count})" for name, count in pitch_counts.most_common(5)
        ))

        # velocity
        print(f"Velocity: {min(velocities)}–{max(velocities)}, "
              f"avg {sum(velocities)/len(velocities):.0f}")

        # duration
        print(f"Note duration: {min(durations):.3f}s–{max(durations):.3f}s, "
              f"avg {sum(durations)/len(durations):.3f}s")

    # ── Lyrics + note density side by side ──
    if lyrics:
        print(f"\n{'─' * 60}")
        print("Lyrics with musical context:")
        print(f"{'─' * 60}")

        timeline = data.get("timeline")
        if timeline:
            for entry in timeline:
                t = f"[{entry['start']:6.1f}–{entry['end']:6.1f}s]"
                lyric = entry["lyric"]
                if "note_count" in entry:
                    pr = entry["pitch_range"]
                    vel = entry.get("avg_velocity", "?")
                    ctx = f"  ♪ {entry['note_count']} notes, {pr[0]}–{pr[1]}, vel~{vel}"
                else:
                    ctx = "  ♪ (sparse)"
                print(f"  {t}  {lyric}{ctx}")
        else:
            for lyric in lyrics:
                print(f"  [{lyric['start']:6.1f}–{lyric['end']:6.1f}s]  {lyric['text']}")

    if notes:
        # density map
        print(f"\n{'─' * 60}")
        print(f"Density (notes per 10s):")
        print(f"{'─' * 60}")
        for t in range(0, int(total_dur) + 1, 10):
            count = len([n for n in notes if t <= n["start"] < t + 10])
            bar = "█" * count
            # annotate with lyrics in this window
            window_lyrics = [l["text"] for l in lyrics
                             if l["start"] >= t and l["start"] < t + 10]
            lyric_hint = f"  ♪ {' '.join(window_lyrics)}" if window_lyrics else ""
            print(f"  {t:3d}–{t+10:3d}s: {count:3d} {bar}{lyric_hint}")

        # pitch map
        print(f"\nPitch map (avg per 15s):")
        for t in range(0, int(total_dur) + 1, 15):
            segment = [n for n in notes if t <= n["start"] < t + 15]
            if segment:
                avg = sum(n["pitch"] for n in segment) / len(segment)
                lo = min(n["pitch"] for n in segment)
                hi = max(n["pitch"] for n in segment)
                print(f"  {t:3d}–{t+15:3d}s: avg={avg:.0f}  [{lo}–{hi}]  "
                      f"n={len(segment)}")

        # chords (simultaneous notes)
        chords = []
        i = 0
        while i < len(notes):
            cluster = [notes[i]]
            j = i + 1
            while j < len(notes) and abs(notes[j]["start"] - notes[i]["start"]) < 0.05:
                cluster.append(notes[j])
                j += 1
            if len(cluster) >= 2:
                names = [n["note_name"] for n in cluster]
                chords.append((cluster[0]["start"], names))
            i = j if j > i + 1 else i + 1

        if chords:
            print(f"\nChords (simultaneous notes): {len(chords)}")
            for t, names in chords[:10]:
                print(f"  {t:6.2f}s  {' + '.join(names)}")
            if len(chords) > 10:
                print(f"  ... and {len(chords) - 10} more")

        # silences
        gaps = []
        for i in range(1, len(notes)):
            gap = notes[i]["start"] - notes[i-1]["end"]
            if gap > 0.5:
                gaps.append((notes[i-1]["end"], notes[i]["start"], gap))
        gaps.sort(key=lambda x: -x[2])

        if gaps:
            print(f"\nLongest silences:")
            for end, start, gap in gaps[:5]:
                print(f"  {end:.1f}s — {start:.1f}s  ({gap:.1f}s)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="whale-listen: convert audio to structured note data for AI"
    )
    parser.add_argument("audio", nargs="?", help="path to audio file (MP3, WAV, etc.)")
    parser.add_argument("-o", "--output", help="output JSON path")
    parser.add_argument("--midi", help="output MIDI path")
    parser.add_argument("--analyze", action="store_true",
                        help="print structural analysis")
    parser.add_argument("--analyze-only", metavar="JSON",
                        help="analyze existing JSON (no conversion)")
    parser.add_argument("--lyrics", action="store_true",
                        help="transcribe vocals with faster-whisper (adds lyrics to JSON)")
    parser.add_argument("--whisper-model", default="base",
                        help="whisper model size: tiny, base, small, medium (default: base)")
    parser.add_argument("--language", default="zh",
                        help="language code for transcription (default: zh, use 'auto' for auto-detect)")

    args = parser.parse_args()

    if args.analyze_only:
        with open(args.analyze_only) as f:
            data = json.load(f)
        analyze(data)
        return

    if not args.audio:
        parser.error("audio file is required (unless using --analyze-only)")

    data = convert(args.audio, args.output, args.midi,
                   do_lyrics=args.lyrics, whisper_model=args.whisper_model,
                   language=args.language)

    if args.analyze:
        analyze(data)


if __name__ == "__main__":
    main()
