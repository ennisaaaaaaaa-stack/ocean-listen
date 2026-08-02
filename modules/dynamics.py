"""Per-note dynamic contour extraction from audio.

Enriches MIDI note data with energy information that basic-pitch's
velocity field can't capture: attack shape, peak position, decay,
and relative intensity. This gives the receiving agent a sense of
how each note was actually played, not just when it started.

For piano music especially, dynamics ARE the music — the difference
between a melody note and an accompaniment note at the same pitch
is often purely in touch (velocity), and basic-pitch flattens that.
"""

import numpy as np


def _note_energy_envelope(y, sr, start, end, hop=512):
    """Extract RMS energy envelope within a note's time window.

    Returns (times, rms_values) arrays, or (None, None) if empty.
    """
    i_start = max(0, int(start * sr / hop) - 1)
    i_end = max(i_start + 1, int(end * sr / hop) + 1)

    # Slice audio and compute frame-level RMS
    s_start = int(start * sr)
    s_end = max(s_start + hop, int(end * sr))
    segment = y[s_start:s_end]
    if len(segment) < hop:
        return None, None

    # Compute RMS per hop
    n_frames = max(1, len(segment) // hop)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = segment[i * hop:(i + 1) * hop]
        if len(frame) > 0:
            rms[i] = np.sqrt(np.mean(frame ** 2))

    times = np.linspace(start, end, n_frames)
    return times, rms


def analyze_dynamics(notes, y, sr=22050, hop=512):
    """Enrich note list with per-note dynamic contour data.

    For each note, adds a 'dynamics' dict with:
        peak_rms:        maximum energy during the note
        mean_rms:        average energy
        attack_slope:    how fast energy rises to peak (rms/sec)
        peak_position:   where peak occurs within note (0=start, 1=end)
        decay_rate:      how fast energy falls after peak
        relative:        z-score relative to all notes (intensity rank)
        dynamic_label:   pp/p/mp/mf/f/ff based on relative intensity

    Also adds phrase-level dynamics at data["phraseDynamics"]:
        time windows with energy trend (rising/falling/stable)
        and crescendo/decrescendo detection.
    """
    if not notes or y is None:
        return notes, None

    print(f"Analyzing dynamics for {len(notes)} notes...")

    # Phase 1: per-note energy extraction
    raw_dynamics = []
    for note in notes:
        start = note.get("start", note.get("start_time", 0))
        end = note.get("end", note.get("end_time", start + 0.5))

        times, rms = _note_energy_envelope(y, sr, start, end, hop)
        if times is None or len(rms) < 2:
            raw_dynamics.append(None)
            continue

        peak_idx = int(np.argmax(rms))
        peak_rms = float(rms[peak_idx])
        mean_rms = float(np.mean(rms))
        dur = end - start

        # Peak position: 0 = at start, 1 = at end
        peak_pos = peak_idx / max(1, len(rms) - 1)

        # Attack slope: energy rise rate before peak
        if peak_idx > 0:
            attack_slope = float(rms[peak_idx] / max(0.001, times[peak_idx] - times[0]))
        else:
            attack_slope = float(rms[0] / max(0.001, dur))

        # Decay: energy drop rate after peak
        if peak_idx < len(rms) - 1:
            tail = rms[peak_idx:]
            decay_rate = float((tail[0] - tail[-1]) / max(0.001, dur * (1 - peak_pos)))
        else:
            decay_rate = 0.0

        raw_dynamics.append({
            "peak_rms": round(peak_rms, 5),
            "mean_rms": round(mean_rms, 5),
            "attack_slope": round(attack_slope, 3),
            "peak_position": round(peak_pos, 2),
            "decay_rate": round(decay_rate, 3),
        })

    # Phase 2: relative intensity (z-score across all notes)
    valid_means = [d["mean_rms"] for d in raw_dynamics if d is not None]
    if valid_means:
        global_mean = np.mean(valid_means)
        global_std = np.std(valid_means) or 0.001

        # Build percentiles for dynamic labels
        sorted_means = sorted(valid_means)
        n = len(sorted_means)
        thresholds = {
            "pp": sorted_means[int(n * 0.10)],
            "p":  sorted_means[int(n * 0.30)],
            "mp": sorted_means[int(n * 0.50)],
            "mf": sorted_means[int(n * 0.70)],
            "f":  sorted_means[int(n * 0.90)],
        }

        for i, dyn in enumerate(raw_dynamics):
            if dyn is None:
                continue
            z = (dyn["mean_rms"] - global_mean) / global_std
            dyn["relative"] = round(float(z), 2)

            # Label based on percentile
            m = dyn["mean_rms"]
            if m <= thresholds["pp"]:
                dyn["dynamic_label"] = "pp"
            elif m <= thresholds["p"]:
                dyn["dynamic_label"] = "p"
            elif m <= thresholds["mp"]:
                dyn["dynamic_label"] = "mp"
            elif m <= thresholds["mf"]:
                dyn["dynamic_label"] = "mf"
            elif m <= thresholds["f"]:
                dyn["dynamic_label"] = "f"
            else:
                dyn["dynamic_label"] = "ff"

    # Attach dynamics back to notes
    for i, note in enumerate(notes):
        if raw_dynamics[i] is not None:
            note["dynamics"] = raw_dynamics[i]

    # Phase 3: phrase-level dynamics (5s windows, 1s step)
    phrase_dynamics = _phrase_dynamics(notes, y, sr, hop)

    return notes, phrase_dynamics


def _phrase_dynamics(notes, y, sr, hop=512, window=5.0, step=1.0):
    """Detect energy trends in 5-second windows.

    Returns list of {start, end, mean_energy, trend, label} where:
        trend: 'rising' / 'falling' / 'stable'
        label: 'crescendo' / 'decrescendo' / 'sustained' / 'settling'
    """
    duration = len(y) / sr
    results = []

    t = 0.0
    while t < duration - step:
        t_end = min(t + window, duration)

        # Get mean energy of notes in this window
        window_notes = [
            n for n in notes
            if n.get("start", 0) >= t and n.get("start", 0) < t_end
        ]

        if not window_notes:
            t += step
            continue

        # Split window into halves, compare energy
        mid = t + (t_end - t) / 2
        first_half = [n for n in window_notes if n.get("start", 0) < mid]
        second_half = [n for n in window_notes if n.get("start", 0) >= mid]

        def avg_energy(note_list):
            vals = [n.get("dynamics", {}).get("mean_rms", 0) for n in note_list if "dynamics" in n]
            return np.mean(vals) if vals else 0

        e1 = avg_energy(first_half)
        e2 = avg_energy(second_half)

        if e1 < 0.001 and e2 < 0.001:
            t += step
            continue

        ratio = e2 / max(e1, 0.001)

        if ratio > 1.15:
            trend, label = "rising", "crescendo"
        elif ratio < 0.87:
            trend, label = "falling", "decrescendo"
        elif e1 > 0.01 and abs(ratio - 1.0) < 0.1:
            trend, label = "stable", "sustained"
        else:
            trend, label = "stable", "settling"

        results.append({
            "start": round(t, 1),
            "end": round(t_end, 1),
            "mean_energy": round(float((e1 + e2) / 2), 5),
            "trend": trend,
            "label": label,
        })

        t += step

    return results
