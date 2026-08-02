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


def run_deep(data, audio_path, cache_dir, pipeline_config=None):
    """Deep listen: mode-aware pipeline.

    pipeline_config comes from classifier.get_pipeline_config().
    If None, defaults to full music mode (backward compat).
    """
    if pipeline_config is None:
        from modules.classifier import get_pipeline_config
        pipeline_config = get_pipeline_config("music")

    audio_type = pipeline_config.get("_type", "music")
    print(f"\n=== Deep listen [{audio_type}] ===")
    print(f"  Pipeline: {pipeline_config['description']}")

    import librosa
    import numpy as np

    if not pipeline_config["run_demucs"]:
        # --- Voice or Solo mode: no Demucs needed ---
        return _run_deep_no_demucs(data, audio_path, pipeline_config)

    # --- Music or Mixed mode: Demucs separation ---
    stems_dir = cache_dir / "stems"
    import modules.stems as stems_mod

    model = pipeline_config.get("demucs_model", "htdemucs_6s")
    tracks = pipeline_config.get("tracks", ("vocals", "drums", "bass", "guitar", "piano", "other"))
    print(f"Separating stems (Demucs {model}, {len(tracks)} tracks)...")
    stems_mod.split(audio_path, stems_dir, model=model)

    print("Building stem timeline...")
    timeline, vocals_y, vocals_rms = stems_mod.build_timeline(stems_dir)
    data["stemTimeline"] = timeline

    # Per-stem MIDI extraction (only for configured tracks)
    if pipeline_config["run_basic_pitch"]:
        import modules.per_stem_notes as psn
        print("Extracting per-stem MIDI notes...")
        stem_notes, all_stem_notes = psn.analyze_all_stems(stems_dir, tracks)
        data["stemNotes"] = {k: v for k, v in stem_notes.items()}
        data["totalStemNotes"] = len(all_stem_notes)

        if stem_notes.get("vocals"):
            print("Detecting vocal parts...")
            parts = psn.detect_vocal_parts(stem_notes["vocals"])
            data["vocalParts"] = parts

        data["unifiedTimeline"] = psn.build_stem_timeline(stem_notes)

    # Voice profile + f0 (on separated vocals track)
    if pipeline_config["run_voice_profile"] and vocals_y is not None and timeline.get("vocals"):
        import modules.voice as voice_mod
        print("Analyzing voice profile...")
        vp = voice_mod.profile(vocals_y, vocals_rms, timeline["vocals"], librosa, np)
        if vp:
            data["voiceProfile"] = vp

    if pipeline_config["run_f0"] and vocals_y is not None:
        try:
            import modules.voice as voice_mod
            print("Extracting f0 trajectory...")
            f0_data = voice_mod.f0_analysis(vocals_y, sr=22050)
            data["vibrato"] = f0_data["vibrato"]
            data["f0Data"] = {"times": f0_data["f0_times"], "values": f0_data["f0_values"]}

            # Voice segmentation
            segments = None
            if pipeline_config.get("run_segment"):
                print("Segmenting voice...")
                segments = voice_mod.segment_voice(
                    f0_data["f0_values"], f0_data["f0_times"], sr=22050)
                if segments:
                    data["voiceSegments"] = segments

            # Timbre analysis
            timbre = None
            if pipeline_config.get("run_timbre"):
                print("Analyzing voice timbre...")
                timbre = voice_mod.voice_timbre_analysis(vocals_y, sr=22050)
                if timbre:
                    data["voiceTimbre"] = timbre

            # Texture profile
            if segments:
                texture = voice_mod.voice_texture_profile(segments, timbre)
                if texture:
                    data["voiceTexture"] = texture

            # Speech analysis
            if pipeline_config.get("run_speech") and segments:
                print("Analyzing speech patterns...")
                speech = voice_mod.speech_analysis(
                    f0_data["f0_values"], f0_data["f0_times"],
                    segments, y=vocals_y, sr=22050)
                if speech:
                    data["speechAnalysis"] = speech

        except Exception as e:
            print(f"f0/timbre analysis failed: {e}")

    data["deepVersion"] = 1
    return data


def _run_deep_no_demucs(data, audio_path, pipeline_config):
    """Deep analysis without Demucs: for voice and solo modes."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # Use structure-level vocal segments for voice analysis
    vocal_segments = data.get("vocalSegments", [])
    if not vocal_segments:
        # Treat entire audio as one segment
        vocal_segments = [[0.0, round(duration, 1)]]

    # Voice profile on original audio
    if pipeline_config["run_voice_profile"]:
        import modules.voice as voice_mod
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        print("Analyzing voice profile (original audio)...")
        vp = voice_mod.profile(y, rms, vocal_segments, librosa, np)
        if vp:
            data["voiceProfile"] = vp

    # f0 trajectory (intonation for speech, pitch for singing)
    segments = None
    if pipeline_config["run_f0"]:
        try:
            import modules.voice as voice_mod
            print("Extracting f0 trajectory...")
            f0_data = voice_mod.f0_analysis(y, sr=22050)
            data["vibrato"] = f0_data["vibrato"]
            data["f0Data"] = {"times": f0_data["f0_times"], "values": f0_data["f0_values"]}

            # Voice segmentation
            if pipeline_config.get("run_segment"):
                print("Segmenting voice...")
                segments = voice_mod.segment_voice(
                    f0_data["f0_values"], f0_data["f0_times"], sr=22050)
                if segments:
                    data["voiceSegments"] = segments

            # Timbre analysis
            timbre = None
            if pipeline_config.get("run_timbre"):
                print("Analyzing voice timbre...")
                timbre = voice_mod.voice_timbre_analysis(y, sr=sr)
                if timbre:
                    data["voiceTimbre"] = timbre

            # Texture profile (combines segments + timbre)
            if pipeline_config.get("run_segment") and segments:
                texture = voice_mod.voice_texture_profile(segments, timbre)
                if texture:
                    data["voiceTexture"] = texture

            # Speech analysis
            if pipeline_config.get("run_speech") and segments:
                print("Analyzing speech patterns...")
                speech = voice_mod.speech_analysis(
                    f0_data["f0_values"], f0_data["f0_times"], segments, y=y, sr=sr)
                if speech:
                    data["speechAnalysis"] = speech

        except Exception as e:
            print(f"f0/timbre analysis failed: {e}")

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
    elif mode == "sensevoice":
        print("\n=== Lyrics (SenseVoice) ===")
        import modules.lyrics_sensevoice as ls
        result = ls.transcribe(str(audio_path), language=language)
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

    # Build aligned timeline
    if "lyrics" in data and data.get("notes"):
        data["timeline"] = _build_aligned_timeline(data["notes"], data["lyrics"])

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
        usage="%(prog)s <audio> [--deep] [--mode auto|music|solo|voice|mixed] [options]"
    )
    parser.add_argument("audio", help="path to audio file")
    parser.add_argument("--deep", action="store_true",
                        help="deep listen: auto-classify and run appropriate pipeline")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "music", "solo", "voice", "mixed"],
                        help="force analysis mode (default: auto-classify)")
    parser.add_argument("--lyric", nargs="?", const="whisper", default=None,
                        help="lyrics source: 'auto' (netease first→whisper fallback), "
                             "'whisper' (local), 'sensevoice' (Alibaba, Chinese+emotion), "
                             "'netease' (search/id/url)")
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
    elif args.lyric == "sensevoice":
        lyric_mode = "sensevoice"
    elif args.lyric == "netease" or (args.lyric and args.lyric not in ("whisper", "auto", "sensevoice")):
        lyric_mode = "netease"
        lyric_value = args.lyric_value or args.lyric

    # --- Shallow listen (always) ---
    data = run_shallow(audio_path, cache_dir, args.force)

    # --- Classification ---
    audio_type = args.mode
    classification = None
    if args.mode == "auto" or args.deep:
        import modules.classifier as classifier_mod
        print("\n=== Pre-classification ===")
        classification = classifier_mod.classify(
            data.get("instruments", {}),
            data,
        )
        print(f"  Detected: {classification['type']} (confidence: {classification['confidence']})")
        print(f"  {classification['reasoning']}")
        data["classification"] = classification
        if args.mode == "auto":
            audio_type = classification["type"]

    # --- Deep listen ---
    if args.deep:
        from modules.classifier import get_pipeline_config
        pipeline_config = get_pipeline_config(audio_type)
        pipeline_config["_type"] = audio_type
        data = run_deep(data, str(audio_path), cache_dir, pipeline_config)

    # --- Lyrics ---
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
