"""Unified report output — text report + JSON.
Combines all Ocean Listen modules into one coherent report.
"""
import json
import pathlib

import pretty_midi

STEM_CN = {
    "vocals": "人声", "drums": "鼓", "bass": "贝斯",
    "guitar": "吉他", "piano": "钢琴", "other": "其它",
}


def mmss(seconds):
    seconds = round(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def print_shallow_report(data):
    """Print the shallow listening report."""
    name = data.get("name", "")
    duration = data.get("duration", 0)
    bpm = data.get("bpm", "?")
    key = data.get("key", "?")

    print(f"\n{'=' * 60}")
    print(f"  Ocean Listen — {name}")
    print(f"  Duration: {round(duration)}s | BPM: {bpm} | Key: {key}")
    print(f"{'=' * 60}")

    # Classification
    cls = data.get("classification")
    if cls:
        print(f"\n  Type: {cls['type'].upper()} (confidence: {cls['confidence']})")
        print(f"  {cls['reasoning']}")

    # Energy curve
    segments = data.get("segments", [])
    if segments:
        peak = max(range(len(segments)), key=lambda i: segments[i]["avgEnergy"])
        low = min(range(len(segments)), key=lambda i: segments[i]["avgEnergy"])
        bar = "".join("▁▂▃▄▅▆▇█"[min(7, int(s["avgEnergy"] / (segments[peak]["avgEnergy"] or 1) * 7.99))]
                      for s in segments)
        print(f"\nEnergy: {bar}")
        print(f"  Peak: {mmss(segments[peak]['start'])}-{mmss(segments[peak]['end'])} | "
              f"Quietest: {mmss(segments[low]['start'])}-{mmss(segments[low]['end'])}")

    # Key progression
    chroma = data.get("chromaBySegment", [])
    if chroma:
        print(f"Key progression: {' → '.join(chroma)}")
    trend = data.get("brightnessTrend", "")
    if trend:
        print(f"Brightness: {trend}")

    # Instruments
    instruments = data.get("instruments", {})
    if instruments:
        print(f"\n— Instruments detected —")
        for inst, segs in sorted(instruments.items(), key=lambda x: x[1][0][0] if x[1] else 9999):
            spans = ", ".join(f"{mmss(s)}-{mmss(e)}" for s, e in segs)
            print(f"  {inst}: {spans}")

    # Notes summary
    notes = data.get("notes", [])
    if notes:
        pitches = [n["pitch"] for n in notes]
        low_note = pretty_midi.note_number_to_name(min(pitches))
        high_note = pretty_midi.note_number_to_name(max(pitches))
        from collections import Counter
        top = Counter(n["note_name"] for n in notes).most_common(5)
        print(f"\n— Notes —")
        print(f"  {len(notes)} notes | Range: {low_note}–{high_note}")
        print(f"  Most common: {', '.join(f'{n}({c})' for n, c in top)}")
        vels = [n["velocity"] for n in notes]
        print(f"  Velocity: {min(vels)}–{max(vels)}, avg {sum(vels)//len(vels)}")

    # Density map
    if notes:
        total_dur = data.get("duration", 0)
        print(f"\n— Density (notes per 10s) —")
        for t in range(0, int(total_dur) + 1, 10):
            count = len([n for n in notes if t <= n["start"] < t + 10])
            bar = "█" * min(count, 50)
            print(f"  {t:4d}–{t+10:4d}s: {count:3d} {bar}")

    # Lyrics
    lyrics = data.get("lyrics", {})
    if lyrics:
        segments = lyrics.get("segments", [])
        if segments:
            print(f"\n— Lyrics ({lyrics.get('language', '?')}) —")
            for seg in segments:
                print(f"  [{seg['start']:6.1f}-{seg['end']:6.1f}s] {seg['text']}")
        elif lyrics.get("lrc"):
            print(f"\n— Lyrics (from {lyrics.get('source', '?')}) —")
            print(lyrics["lrc"].rstrip())


def print_deep_report(data):
    """Print the deep listening report (stems + voice)."""
    stem_timeline = data.get("stemTimeline", {})
    if stem_timeline:
        print(f"\n{'─' * 60}")
        print("— Stem timeline (Demucs 6-track) —")
        print(f"{'─' * 60}")
        for track, segs in sorted(stem_timeline.items(),
                                   key=lambda x: x[1][0][0] if x[1] else 9999):
            cn = STEM_CN.get(track, track)
            if segs:
                spans = ", ".join(f"{mmss(s)}-{mmss(e)}" for s, e in segs)
            else:
                spans = "inactive"
            print(f"  {cn:4s} ({track:7s}): {spans}")

    # Per-stem notes
    stem_notes = data.get("stemNotes", {})
    if stem_notes:
        print(f"\n— Per-stem MIDI notes —")
        for stem, notes in stem_notes.items():
            cn = STEM_CN.get(stem, stem)
            if notes:
                pitches = [n["pitch"] for n in notes]
                lo = pretty_midi.note_number_to_name(min(pitches))
                hi = pretty_midi.note_number_to_name(max(pitches))
                print(f"  {cn:4s}: {len(notes)} notes, {lo}–{hi}")
            else:
                print(f"  {cn:4s}: 0 notes")

    # Vocal parts
    vocal_parts = data.get("vocalParts", [])
    if vocal_parts:
        print(f"\n— Vocal parts —")
        for part in vocal_parts:
            pr = part.get("pitch_range")
            pr_str = f" {pretty_midi.note_number_to_name(pr[0])}–{pretty_midi.note_number_to_name(pr[1])}" if pr else ""
            print(f"  {part['label']}: {part['note_count']} notes{pr_str} ({part['likely_gender']})")

    # Voice profile
    vp = data.get("voiceProfile", {})
    if vp:
        print(f"\n— Voice profile —")
        soft = vp.get("softWindow", {})
        burst = vp.get("burstWindow", {})
        if soft:
            print(f"  Soft (at {mmss(soft.get('start', 0))}): "
                  f"breath {soft.get('breathNoiseRatio', 0)*100:.1f}% | "
                  f"air {soft.get('airRatio', 0)*100:.1f}%")
        if burst:
            print(f"  Burst (at {mmss(burst.get('start', 0))}): "
                  f"breath {burst.get('breathNoiseRatio', 0)*100:.1f}% | "
                  f"air {burst.get('airRatio', 0)*100:.1f}%")
        lr = vp.get("loudnessRatio")
        if lr:
            print(f"  Loudness ratio: {lr}x")
        tr = vp.get("tailReverb")
        if tr is not None:
            print(f"  Reverb tail: {tr:.2f}s")

    # Vibrato
    vib = data.get("vibrato", {})
    if vib:
        if vib.get("detected"):
            print(f"  Vibrato: {vib['average_rate']} Hz, depth {vib['average_depth_cents']} cents")
        else:
            print(f"  Vibrato: minimal/none (modern straight-tone style)")

    # Unified stem timeline
    unified = data.get("unifiedTimeline", [])
    if unified:
        print(f"\n— Unified timeline —")
        for entry in unified[:6]:  # first 6 windows
            t_s = entry["start"]
            t_e = entry["end"]
            active = entry.get("stems", {})
            parts = []
            for stem, info in sorted(active.items()):
                cn = STEM_CN.get(stem, stem)
                parts.append(f"{cn}({info['note_count']})")
            print(f"  {mmss(t_s)}-{mmss(t_e)}: {' | '.join(parts)}")


def save_json(data, output_path):
    """Save full analysis to JSON."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON -> {path}")
