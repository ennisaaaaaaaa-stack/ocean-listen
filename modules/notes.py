"""MIDI note extraction via basic-pitch."""
import pathlib
import json

import scipy.signal
from scipy.signal.windows import gaussian as _gaussian
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = _gaussian

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi


def find_model():
    onnx = pathlib.Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
    if onnx.exists():
        return onnx
    return ICASSP_2022_MODEL_PATH


def extract_notes(audio_path, output_midi=None):
    """Run basic-pitch on an audio file, return list of note dicts."""
    audio = pathlib.Path(audio_path)
    model = find_model()
    print(f"Listening to {audio.name}...")

    model_output, midi_data, note_events = predict(str(audio), model)

    if output_midi:
        midi_data.write(str(output_midi))
        print(f"MIDI -> {output_midi} ({len(note_events)} note events)")
    elif output_midi is None:
        midi_path = audio.with_suffix(".mid")
        midi_data.write(str(midi_path))
        print(f"MIDI -> {midi_path} ({len(note_events)} note events)")

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
    print(f"Extracted {len(notes)} notes")
    return notes


def extract_notes_from_stem(stem_path, stem_name=""):
    """Extract notes from a single Demucs stem."""
    prefix = f"[{stem_name}] " if stem_name else ""
    print(f"{prefix}Extracting notes from stem...")
    notes = extract_notes(stem_path, output_midi=False)
    return notes
