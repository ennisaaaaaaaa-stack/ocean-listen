"""Lyrics transcription via faster-whisper (local, offline)."""


def transcribe(audio_path, model_size="small", language="auto"):
    """Transcribe vocals using faster-whisper.
    
    Returns list of {text, start, end} dicts.
    """
    from faster_whisper import WhisperModel

    print(f"Loading whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing vocals in {audio_path}...")
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
    return {"segments": lyrics, "language": info.language}
