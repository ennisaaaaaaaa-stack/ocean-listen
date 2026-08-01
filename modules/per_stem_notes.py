"""Per-stem MIDI extraction and vocal multi-part detection.
This is Ocean Listen's core innovation — not present in either parent project.
"""
import pathlib
import json
import pretty_midi

# Male/female pitch ranges (MIDI note numbers)
# Male fundamental: ~C2(36) to F3(53), Female: ~E3(52) to D4(50)
MALE_RANGE = (36, 55)   # C2 to G3
FEMALE_RANGE = (52, 72) # E3 to C5


def extract_stem_notes(stem_audio_path, stem_name):
    """Run basic-pitch on a single Demucs stem."""
    from modules.notes import extract_notes
    print(f"[{stem_name}] Extracting MIDI from stem...")
    notes = extract_notes(str(stem_audio_path), output_midi=False)
    for n in notes:
        n["stem"] = stem_name
    return notes


def analyze_all_stems(stems_dir, track_names):
    """Extract MIDI notes from each separated stem.
    
    Returns: {track_name: [note_dicts], ...} and combined list.
    """
    stem_notes = {}
    all_notes = []

    for track in track_names:
        stem_path = pathlib.Path(stems_dir) / f"{track}.mp3"
        if not stem_path.exists():
            print(f"[{track}] Stem not found, skipping")
            stem_notes[track] = []
            continue
        notes = extract_stem_notes(stem_path, track)
        stem_notes[track] = notes
        all_notes.extend(notes)
        print(f"[{track}] {len(notes)} notes, "
              f"pitch range: {min((n['pitch'] for n in notes), default=0)}-"
              f"{max((n['pitch'] for n in notes), default=0)}")

    return stem_notes, all_notes


def detect_vocal_parts(vocal_notes):
    """Cluster vocal notes by pitch to detect multiple voice parts.
    
    Uses simple k-means-like clustering on pitch histogram.
    Returns list of detected parts with pitch range and likely gender.
    """
    if len(vocal_notes) < 20:
        return [{"label": "single", "pitch_range": None,
                 "likely_gender": "unknown", "note_count": len(vocal_notes)}]

    pitches = [n["pitch"] for n in vocal_notes]

    # Build pitch histogram (semitone resolution)
    pitch_hist = {}
    for p in pitches:
        pitch_hist[p] = pitch_hist.get(p, 0) + 1

    # Find peaks by looking at binned distribution
    # Try splitting at the male/female boundary zone (G2 to G3 = 43 to 55)
    low_notes = [p for p in pitches if p <= 50]   # <= D3
    high_notes = [p for p in pitches if p > 50]    # > D3

    parts = []

    if low_notes and high_notes:
        # Two distinct groups
        low_ratio = len(low_notes) / len(pitches)
        high_ratio = len(high_notes) / len(pitches)

        if 0.15 < low_ratio and 0.15 < high_ratio:
            # Both groups are significant — likely two parts
            parts.append({
                "label": "low",
                "pitch_range": [min(low_notes), max(low_notes)],
                "note_count": len(low_notes),
                "likely_gender": "male" if max(low_notes) <= 55 else "unknown",
            })
            parts.append({
                "label": "high",
                "pitch_range": [min(high_notes), max(high_notes)],
                "note_count": len(high_notes),
                "likely_gender": "female" if min(high_notes) >= 52 else "unknown",
            })
            return parts

    # Single part
    parts.append({
        "label": "single",
        "pitch_range": [min(pitches), max(pitches)],
        "note_count": len(pitches),
        "likely_gender": _guess_gender(pitches),
    })
    return parts


def _guess_gender(pitches):
    """Guess gender from average pitch."""
    avg = sum(pitches) / len(pitches)
    if avg <= 50:
        return "likely male"
    elif avg >= 57:
        return "likely female"
    else:
        return "ambiguous"


def build_stem_timeline(notes_per_stem, lyrics=None):
    """Build a unified timeline showing what each instrument is doing.
    
    For each time window, shows: which stems are active, note density per stem,
    and lyrics if available.
    """
    all_stems = list(notes_per_stem.keys())
    timeline = {}

    for stem, notes in notes_per_stem.items():
        if not notes:
            continue
        # 10-second windows
        max_t = max(n["end"] for n in notes)
        for t_start in range(0, int(max_t) + 1, 10):
            t_end = t_start + 10
            window_notes = [n for n in notes if n["start"] < t_end and n["end"] > t_start]
            if window_notes:
                key = t_start
                if key not in timeline:
                    timeline[key] = {"start": t_start, "end": t_end, "stems": {}}
                pitches = [n["pitch"] for n in window_notes]
                timeline[key]["stems"][stem] = {
                    "note_count": len(window_notes),
                    "pitch_range": [min(pitches), max(pitches)],
                    "avg_velocity": round(sum(n["velocity"] for n in window_notes) / len(window_notes)),
                }

    return [timeline[k] for k in sorted(timeline.keys())]
