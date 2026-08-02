"""Pre-classifier: determine audio type before deep analysis.

Uses structure-level signals (percussion ratio, vocal coverage) as primary,
PANNs as secondary (unreliable on mixed/mastered audio).

Classification:
  music  — rhythmic content + moderate vocals → Demucs 6-track + per-stem MIDI
  solo   — single instrument, no percussion, no vocals → skip Demucs
  voice  — vocal dominant, no percussion → f0 + voice profile
  mixed  — vocals + light backing (percussion present) → Demucs 4-track

Key insight: PANNs instrument detection fails on mixed audio (often detects
only "speech" in vocal melodies). Percussion ratio from HPSS is the most
reliable music/non-music discriminator.
"""

# --- Thresholds ---
# Above this percussive ratio → has rhythmic transients → music territory
PERC_MUSIC_THRESHOLD = 0.2
# Above this vocal coverage → vocals dominate the audio
VOCAL_DOMINANT_THRESHOLD = 0.5
# Above this vocal coverage → vocals are present
VOCAL_PRESENCE_THRESHOLD = 0.15

VOCAL_GROUPS = {"vocals", "speech"}
INSTRUMENT_GROUPS = {
    "guitar", "bass", "drums", "piano", "synth",
    "strings", "brass", "organ", "whistle",
}


def classify(instruments_data, structure_data):
    """Classify audio into one of: music, solo, voice, mixed.

    Args:
        instruments_data: output of instruments.detect() (dict with _confidence)
        structure_data: output of structure.analyze() (has percussiveRatio, vocalCoverage)

    Returns:
        dict with type, confidence, reasoning, details
    """
    # === Primary signals (from structure — reliable on mixed audio) ===
    perc_ratio = structure_data.get("percussiveRatio", 0)
    vocal_coverage = structure_data.get("vocalCoverage", 0)
    bpm = structure_data.get("bpm", 0)
    duration = structure_data.get("duration", 1)

    # === Secondary signals (from PANNs — useful when clear, unreliable on mixed) ===
    segments = {
        k: v for k, v in instruments_data.items() if not k.startswith("_")
    }
    confidence_raw = instruments_data.get("_confidence", {})

    has_panns_speech = bool(segments.get("speech"))
    has_panns_singing = bool(segments.get("vocals"))

    # Count PANNs instrument groups (may be incomplete for mixed music)
    panns_instruments = [
        g for g in segments
        if g in INSTRUMENT_GROUPS and segments[g]
    ]

    # PANNs speech coverage (low for music, high for actual speech)
    speech_segs = segments.get("speech", [])
    speech_time = sum(e - s for s, e in speech_segs) if speech_segs else 0
    speech_coverage = speech_time / duration if duration else 0

    # === Decision tree ===
    has_percussion = perc_ratio > PERC_MUSIC_THRESHOLD
    has_vocals = vocal_coverage > VOCAL_PRESENCE_THRESHOLD or has_panns_speech or has_panns_singing
    vocals_dominant = vocal_coverage > VOCAL_DOMINANT_THRESHOLD

    if has_percussion and vocals_dominant:
        # Vocals dominate but there's rhythmic backing → podcast, rap, talk over music
        audio_type = "mixed"
        reasoning = (f"Vocals dominate ({vocal_coverage:.0%}) with rhythmic backing "
                     f"(perc={perc_ratio:.2f})")
        conf_score = 0.8
    elif has_percussion and not panns_instruments and speech_coverage > 0.2:
        # Percussion present but PANNs found ONLY voice (speech/vocals), no instruments
        # → percussion is from consonants/breathing, not drums
        # This catches raw voice recordings that have "noisy" articulation
        audio_type = "voice"
        reasoning = (f"Voice-only detected (speech_cov={speech_coverage:.0%}), "
                     f"percussion from articulation (perc={perc_ratio:.2f}), "
                     f"no instruments found")
        conf_score = 0.85
    elif has_percussion:
        # Has rhythmic transients → almost certainly music
        # PANNs can't see instruments in mixed audio, but structure says music
        audio_type = "music"
        if panns_instruments:
            reasoning = (f"{len(panns_instruments)} instruments detected ({panns_instruments}), "
                         f"perc={perc_ratio:.2f}")
        else:
            reasoning = (f"Rhythmic content (perc={perc_ratio:.2f}), "
                         f"likely mixed/mastered music (PANNs couldn't parse instruments)")
        conf_score = 0.85
    elif has_vocals:
        # No percussion, but vocals present → speech or a cappella
        audio_type = "voice"
        reasoning = (f"Voice dominant (vocal_cov={vocal_coverage:.0%}, "
                     f"speech_cov={speech_coverage:.0%}), low percussion (perc={perc_ratio:.2f})")
        conf_score = 0.8
    elif panns_instruments:
        # PANNs detected instruments but no percussion or vocals → solo instrument
        audio_type = "solo"
        reasoning = (f"Single instrument ({panns_instruments}), "
                     f"no percussion or vocals (perc={perc_ratio:.2f})")
        conf_score = 0.85
    elif bpm > 60:
        # No percussion detected but has structured rhythm → soft music
        audio_type = "music"
        reasoning = (f"Structured audio (BPM={bpm}) but low percussion (perc={perc_ratio:.2f}) "
                     f"— possibly soft/ambient music")
        conf_score = 0.5
    else:
        # No clear signals — fallback to solo (basic-pitch will still try to extract notes)
        audio_type = "solo"
        reasoning = (f"No clear signals (perc={perc_ratio:.2f}, vocal_cov={vocal_coverage:.0%}, "
                     f"bpm={bpm})")
        conf_score = 0.3

    return {
        "type": audio_type,
        "confidence": conf_score,
        "reasoning": reasoning,
        "details": {
            "instruments": panns_instruments,
            "has_vocals": has_vocals,
            "has_speech": has_panns_speech,
            "has_singing": has_panns_singing,
            "instrument_count": len(panns_instruments),
            "percussiveRatio": perc_ratio,
            "vocalCoverage": vocal_coverage,
            "speechCoverage": round(speech_coverage, 3),
            "bpm": bpm,
        },
    }


# --- Pipeline routing ---

def get_pipeline_config(audio_type):
    """Return pipeline config for a given audio type."""
    configs = {
        "music": {
            "run_demucs": True,
            "demucs_model": "htdemucs_6s",
            "tracks": ("vocals", "drums", "bass", "guitar", "piano", "other"),
            "run_basic_pitch": True,
            "run_voice_profile": True,
            "run_f0": True,
            "description": "Full music analysis: 6-track separation + per-stem MIDI",
        },
        "solo": {
            "run_demucs": False,
            "demucs_model": None,
            "tracks": (),
            "run_basic_pitch": True,
            "run_voice_profile": False,
            "run_f0": False,
            "description": "Solo instrument: direct basic-pitch, no separation needed",
        },
        "voice": {
            "run_demucs": False,
            "demucs_model": None,
            "tracks": (),
            "run_basic_pitch": False,
            "run_voice_profile": True,
            "run_f0": True,
            "description": "Voice analysis: intonation + voice profile + non-vocal sounds",
        },
        "mixed": {
            "run_demucs": True,
            "demucs_model": "htdemucs",
            "tracks": ("vocals", "other"),
            "run_basic_pitch": True,
            "run_voice_profile": True,
            "run_f0": True,
            "description": "Mixed: vocal/accompaniment split + voice analysis",
        },
    }
    return configs.get(audio_type, configs["music"])
