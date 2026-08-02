"""Harmonic/overtone filter for per-stem MIDI notes.

basic-pitch detects harmonics and cross-stem bleed as independent notes.
This module cleans up the raw output to produce a usable melody line.

Three-stage filter:
1. Pitch range gate — each instrument has a realistic range; notes outside are bleed.
2. Overlap dedup — when notes overlap in time, keep the strongest (fundamental > harmonic).
3. Duration gate — ultra-short notes (< 50ms) are detection artifacts.
"""
import numpy as np

# Realistic pitch ranges per stem (MIDI note numbers)
# Calibrated from Soaked/Toxic/Whoregasm stem data, 2026-08-02
STEM_RANGES = {
    "vocals":  (48, 84),    # C3 to C6 — covers male low to female high
    "bass":    (28, 52),    # E1 to E3
    "guitar":  (38, 76),    # D2 to E5
    "piano":   (33, 96),    # A1 to C7 — piano genuinely has wide range
    "drums":   (0, 127),    # unpitched — keep everything
    "other":   (36, 84),    # C2 to C6 — conservative catch-all
}

# Minimum note duration in seconds
MIN_DUR_S = 0.05

# Overlap threshold: if two notes overlap by more than this fraction of the
# shorter note's duration, they're considered competing and we keep the louder one
OVERLAP_THRESHOLD = 0.3


def filter_stem_notes(notes, stem_name):
    """Filter a single stem's note list to remove harmonics and artifacts.

    Args:
        notes: list of note dicts with keys: pitch, start, end, duration, velocity
        stem_name: stem type (vocals, bass, guitar, piano, drums, other)

    Returns:
        (filtered_notes, stats) where stats has counts of what was removed.
    """
    stats = {"input": len(notes), "range_removed": 0, "duration_removed": 0, "overlap_removed": 0}

    if not notes:
        return [], stats

    # --- Stage 1: pitch range gate ---
    lo, hi = STEM_RANGES.get(stem_name, (0, 127))
    stage1 = [n for n in notes if lo <= n["pitch"] <= hi]
    stats["range_removed"] = len(notes) - len(stage1)

    # --- Stage 2: duration gate ---
    stage2 = [n for n in stage1 if n.get("duration", n.get("end", 0) - n.get("start", 0)) >= MIN_DUR_S]
    stats["duration_removed"] = len(stage1) - len(stage2)

    # --- Stage 3: overlap dedup ---
    if not stage2:
        stats["output"] = 0
        return [], stats

    # Sort by start time
    stage2.sort(key=lambda n: n["start"])

    # For drums, skip overlap dedup — percussive hits can legitimately overlap
    if stem_name == "drums":
        stats["output"] = len(stage2)
        return stage2, stats

    filtered = []
    for note in stage2:
        # Check against already-kept notes
        displaced = False
        for kept in reversed(filtered):
            # Only check recent notes (within 2 seconds)
            if kept["start"] > note["start"] + 2:
                break

            # Calculate overlap
            overlap_start = max(note["start"], kept["start"])
            overlap_end = min(note["end"], kept["end"])
            overlap = overlap_end - overlap_start

            if overlap <= 0:
                continue

            shorter_dur = min(
                note.get("duration", note["end"] - note["start"]),
                kept.get("duration", kept["end"] - kept["start"])
            )

            if shorter_dur <= 0:
                continue

            overlap_ratio = overlap / shorter_dur

            if overlap_ratio > OVERLAP_THRESHOLD:
                # They overlap significantly — keep the louder one
                if note["velocity"] > kept["velocity"]:
                    # Replace kept with current note
                    filtered.remove(kept)
                    filtered.append(note)
                    displaced = True
                    break
                else:
                    # Current note is quieter — skip it
                    displaced = True
                    break
        else:
            filtered.append(note)

        if not displaced:
            if note not in filtered:
                filtered.append(note)

    # Re-sort
    filtered.sort(key=lambda n: n["start"])
    stats["overlap_removed"] = len(stage2) - len(filtered)
    stats["output"] = len(filtered)

    return filtered, stats


def filter_all_stems(stem_notes_dict):
    """Filter notes for all stems in a dict.

    Args:
        stem_notes_dict: {stem_name: [note_dicts], ...}

    Returns:
        (filtered_dict, stats_dict)
    """
    result = {}
    all_stats = {}
    for stem, notes in stem_notes_dict.items():
        filtered, stats = filter_stem_notes(notes, stem)
        result[stem] = filtered
        all_stats[stem] = stats
    return result, all_stats


def print_filter_report(all_stats):
    """Print a human-readable summary of what was filtered."""
    print("\n  Harmonic Filter Report:")
    print("  %-10s %6s %6s %6s %6s %6s" % ("Stem", "Input", "Range", "Short", "Ovrlp", "Output"))
    print("  " + "-" * 44)
    for stem, s in all_stats.items():
        print("  %-10s %6d %6d %6d %6d %6d" % (
            stem, s["input"], s["range_removed"],
            s["duration_removed"], s["overlap_removed"], s["output"]))
