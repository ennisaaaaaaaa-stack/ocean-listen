#!/usr/bin/env python3
"""
Ocean Listen / 听海
Let your AI hear music — MIDI notes + instrument recognition + stem separation + voice profile + lyrics.

A merger of:
  - whale-listen (migratorywhale, MIT): MIDI note extraction + whisper lyrics
  - Tinggu 听骨 (SeithAsync, MIT): PANNs instruments + Demucs stems + voice profile
  - NEW: per-stem MIDI extraction + vocal multi-part detection

Usage:
  python ocean.py song.mp3                              # shallow listen
  python ocean.py song.mp3 --deep                       # + stem separation + voice
  python ocean.py song.mp3 --lyric auto                     # best lyrics: netease→whisper
  python ocean.py song.mp3 --lyric whisper --language en     # whisper only
  python ocean.py song.mp3 --lyric "song artist"             # netease search
  python ocean.py song.mp3 --deep --lyric auto               # full experience
"""
import argparse
import json
import pathlib
import sys

# Ensure modules/ is importable
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def run_shallow(audio_path, cache_dir, force=False):
    """Shallow listen: structure + instruments + notes."""
    import modules.structure as structure
    import modules.notes as notes_mod

    print("=== Shallow listen ===")
    data = {}

    # Structure (BPM, key, energy, bands, brightness)
    print("Analyzing structure...")
    struct = structure.analyze(audio_path)
    data.update(struct)

    # MIDI notes
    print("Extracting MIDI notes...")
    midi_output = cache_dir / f"{pathlib.Path(audio_path).stem}.mid"
    note_list = notes_mod.extract_notes(str(audio_path), output_midi=str(midi_output))
    data["notes"] = note_list
    data["total_notes"] = len(note_list)

    # Instruments (PANNs) — optional, needs heavy dep
    try:
        print("Detecting instruments (PANNs)...")
        import modules.instruments as instruments_mod
        inst = instruments_mod.detect(audio_path)
        data["instruments"] = inst
    except ImportError:
        print("PANNs not installed, skipping instrument detection. pip install panns-inference")
        data["instruments"] = {}
    except Exception as e:
        print(f"Instrument detection failed: {e}")
        data["instruments"] = {}

    # Spectrogram
    try:
        print("Generating spectrogram...")
        import modules.visualize as viz
        img_path = viz.generate(audio_path, data, cache_dir)
        data["spectrogram"] = img_path
    except Exception as e:
        print(f"Spectrogram generation failed: {e}")

    data["name"] = pathlib.Path(audio_path).stem
    data["sourcePath"] = str(audio_path)
    data["shallowVersion"] = 1

    return data


def run_deep(data, audio_path, cache_dir, force=False):
    """Deep listen: stem separation + per-stem notes + voice profile."""
    print("\n=== Deep listen ===")

    stems_dir = cache_dir / "stems"

    # Demucs separation
    import modules.stems as stems_mod
    print("Separating stems (Demucs 6-track)...")
    stems_mod.split(audio_path, stems_dir)

    # Per-track timeline
    import librosa
    import numpy as np
    print("Building stem timeline...")
    timeline, vocals_y, vocals_rms = stems_mod.build_timeline(stems_dir)
    data["stemTimeline"] = timeline

    # Per-stem MIDI extraction (Ocean Listen's innovation)
    import modules.per_stem_notes as psn
    TRACKS = ("vocals", "drums", "bass", "guitar", "piano", "other")
    print("Extracting per-stem MIDI notes...")
    stem_notes, all_stem_notes = psn.analyze_all_stems(stems_dir, TRACKS)
    data["stemNotes"] = {k: v for k, v in stem_notes.items()}
    data["totalStemNotes"] = len(all_stem_notes)

    # Vocal multi-part detection
    if stem_notes.get("vocals"):
        print("Detecting vocal parts...")
        parts = psn.detect_vocal_parts(stem_notes["vocals"])
        data["vocalParts"] = parts

    # Unified timeline (what each instrument is doing per 10s window)
    data["unifiedTimeline"] = psn.build_stem_timeline(stem_notes)

    # Voice profile
    if vocals_y is not None and timeline.get("vocals"):
        import modules.voice as voice_mod
        print("Analyzing voice profile...")
        vp = voice_mod.profile(vocals_y, vocals_rms, timeline["vocals"], librosa, np)
        if vp:
            data["voiceProfile"] = vp

        # f0 + vibrato
        try:
            print("Extracting f0 trajectory...")
            f0_data = voice_mod.f0_analysis(vocals_y, sr=22050)
            data["vibrato"] = f0_data["vibrato"]
            data["f0Data"] = {"times": f0_data["f0_times"], "values": f0_data["f0_values"]}
        except Exception as e:
            print(f"f0 analysis failed: {e}")

    data["deepVersion"] = 1
    return data


def run_lyrics(data, audio_path, mode, value, language="auto", whisper_model="small"):
    """Attach lyrics from whisper, netease, or auto (netease-first)."""
    if mode == "auto":
        return _run_lyrics_auto(data, audio_path, value, language, whisper_model)

    if mode == "whisper":
        print("\n=== Lyrics (whisper) ===")
        import modules.lyrics_whisper as lw
        result = lw.transcribe(str(audio_path), model_size=whisper_model, language=language)
        data["lyrics"] = result
    elif mode == "netease":
        print("\n=== Lyrics (NetEase) ===")
        _attach_netease(data, audio_path, value, fallback_whisper=True,
                        language=language, whisper_model=whisper_model)

    # Build aligned timeline
    if "lyrics" in data and data.get("notes"):
        data["timeline"] = _build_aligned_timeline(data["notes"], data["lyrics"])

    return data


def _run_lyrics_auto(data, audio_path, search_term, language, whisper_model):
    """Auto mode: try NetEase first, fall back to whisper."""
    import modules.lyrics_netease as ln
    import pathlib

    # Build search term from filename if not provided
    if not search_term:
        stem = pathlib.Path(audio_path).stem
        search_term = stem.replace("_", " ")

    print(f"\n=== Lyrics (auto) ===")
    print(f"Searching NetEase for '{search_term}'...")

    audio_dur = ln._local_duration_s(str(audio_path))
    best_id = None
    best_lines = None
    best_lrc = None
    duration_mismatch = False

    try:
        candidates = ln._search(search_term)
        if not candidates:
            raise ln.LyricError(f"No results for '{search_term}'")

        # Pick best duration match, but keep closest even if mismatch
        hits = [c for c in candidates
                if abs(c["duration_ms"] / 1000 - audio_dur) <= ln.DURATION_TOLERANCE_S]

        if hits:
            chosen = hits[0]
        else:
            # Duration mismatch — use top result but flag it
            chosen = candidates[0]
            duration_mismatch = True
            orig_dur = chosen["duration_ms"] / 1000
            print(f"  Duration mismatch: audio={audio_dur:.0f}s, "
                  f"netease={orig_dur:.0f}s — using best match "
                  f"(id {chosen['id']}), timestamps will be rescaled")

        lrc, tlrc = ln._lyric(chosen["id"])
        lines = ln.parse_lrc(lrc)

        if not lines:
            raise ln.LyricError(f"No timestamped lyrics for id {chosen['id']}")

        best_id = chosen["id"]
        best_lines = lines
        best_lrc = lrc

    except Exception as e:
        print(f"  NetEase unavailable ({e}), falling back to whisper...")

    if best_lines is not None:
        # Rescale timestamps if duration mismatch
        if duration_mismatch and best_lines:
            orig_dur = max(best_lines[-1][0], 1)
            scale = audio_dur / orig_dur if orig_dur < audio_dur else 1.0
            best_lines = [[round(t * scale, 3), txt] for t, txt in best_lines]

        data["lyrics"] = {
            "source": f"netease:{best_id}",
            "segments": [{"text": t, "start": s, "end": s + 5}
                         for s, t in best_lines],
            "lrc": best_lrc or "",
            "language": "netease",
        }
        print(f"  Lyrics from NetEase (id {best_id}, "
              f"{'rescaled' if duration_mismatch else 'exact match'})")
    else:
        # Fallback: whisper
        print("  Falling back to whisper transcription...")
        import modules.lyrics_whisper as lw
        result = lw.transcribe(str(audio_path), model_size=whisper_model, language=language)
        data["lyrics"] = result

    return data


def _attach_netease(data, audio_path, value, fallback_whisper=False,
                    language="auto", whisper_model="small"):
    """Attach NetEase lyrics with optional whisper fallback."""
    import modules.lyrics_netease as ln
    try:
        result = ln.obtain_lyric(value, audio_path=audio_path)
        data["lyrics"] = {
            "source": result["source"],
            "segments": [{"text": t, "start": s, "end": s + 5}
                         for s, t in result.get("lines", [])],
            "lrc": result.get("lrc", ""),
            "language": "netease",
        }
    except Exception as e:
        print(f"NetEase lyrics failed: {e}")
        if fallback_whisper:
            print("Falling back to whisper...")
            import modules.lyrics_whisper as lw
            data["lyrics"] = lw.transcribe(str(audio_path),
                                           model_size=whisper_model, language=language)


def _build_aligned_timeline(notes, lyrics):
    """Merge notes and lyrics into aligned timeline."""
    import pretty_midi
    segments = lyrics.get("segments", []) if isinstance(lyrics, dict) else lyrics
    if not segments:
        return []

    timeline = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        active = [n for n in notes if n["start"] < seg_end and n["end"] > seg_start]

        entry = {"start": seg_start, "end": seg_end, "lyric": seg["text"]}
        if active:
            pitches = [n["pitch"] for n in active]
            entry["note_count"] = len(active)
            entry["pitch_range"] = [
                pretty_midi.note_number_to_name(min(pitches)),
                pretty_midi.note_number_to_name(max(pitches)),
            ]
            entry["avg_velocity"] = round(sum(n["velocity"] for n in active) / len(active))
        timeline.append(entry)

    return timeline


def main():
    parser = argparse.ArgumentParser(
        description="Ocean Listen / 听海 — let your AI hear music",
        usage="%(prog)s <audio> [--deep] [--lyric whisper|netease] [options]"
    )
    parser.add_argument("audio", help="path to audio file")
    parser.add_argument("--deep", action="store_true",
                        help="deep listen: Demucs stem separation + voice profile + per-stem MIDI")
    parser.add_argument("--lyric", nargs="?", const="whisper", default=None,
                        help="lyrics source: 'auto' (netease first→whisper fallback), "
                             "'whisper' (local), 'netease' (search/id/url)")
    parser.add_argument("--lyric-value", default=None,
                        help="search term for auto/netease: song ID, URL, or 'song artist' "
                             "(auto-mode defaults to filename)")
    parser.add_argument("--language", default="auto",
                        help="language for whisper (default: auto)")
    parser.add_argument("--whisper-model", default="small",
                        help="whisper model: tiny, base, small, medium (default: small)")
    parser.add_argument("--output", "-o", help="output JSON path")
    parser.add_argument("--cache-dir", default=None,
                        help="cache directory (default: ocean_cache/<filename>)")
    parser.add_argument("--force", action="store_true",
                        help="ignore cache, recompute everything")

    args = parser.parse_args()

    audio_path = pathlib.Path(args.audio).resolve()
    if not audio_path.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else \
        pathlib.Path.cwd() / "ocean_cache" / audio_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or str(cache_dir / f"{audio_path.stem}.json")

    # Determine lyric mode
    lyric_mode = None
    lyric_value = None
    if args.lyric == "auto":
        lyric_mode = "auto"
        lyric_value = args.lyric_value
    elif args.lyric == "whisper":
        lyric_mode = "whisper"
    elif args.lyric == "netease" or (args.lyric and args.lyric not in ("whisper", "auto")):
        lyric_mode = "netease"
        lyric_value = args.lyric_value or args.lyric

    # Run analysis
    data = run_shallow(audio_path, cache_dir, args.force)

    if args.deep:
        data = run_deep(data, str(audio_path), cache_dir, args.force)

    if lyric_mode:
        data = run_lyrics(data, str(audio_path), lyric_mode, lyric_value,
                          args.language, args.whisper_model)

    data["name"] = audio_path.stem

    # Print report
    import modules.report as report
    report.print_shallow_report(data)
    if args.deep:
        report.print_deep_report(data)
    report.save_json(data, output_path)

    print(f"\nDone. Full analysis: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
