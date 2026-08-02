"""Lyrics transcription via SenseVoice (Alibaba FunASR).

Strengths vs Whisper:
  - Better Chinese accuracy
  - Emotion detection (HAPPY / SAD / ANGRY / NEUTRAL)
  - Audio event detection (Speech / Music / BGM / Noise)
  - Language identification (zh / en / ja / ko / yue)

Output format matches lyrics_whisper for drop-in replacement.
"""

import re

# Tag patterns in SenseVoice output: <|zh|><|SAD|><|Speech|><|withitn|>text
_TAG_RE = re.compile(r'<\|([^|]+)\|>')

# Emotion codes
_EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}
# Event types
_EVENTS = {"Speech", "Music", "BGM", "Noise"}
# Languages
_LANG_MAP = {
    "zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue",
    "auto": "auto",
}


def _parse_tags(raw_text):
    """Extract metadata tags from SenseVoice output."""
    tags = _TAG_RE.findall(raw_text)
    clean = _TAG_RE.sub('', raw_text).strip()

    language = "auto"
    emotion = None
    event = None
    itn = None

    for tag in tags:
        tag_upper = tag.upper()
        if tag_upper in _EMOTIONS:
            emotion = tag_upper.lower()
        elif tag_upper in _EVENTS:
            event = tag_upper.lower()
        elif tag.lower() in _LANG_MAP:
            language = _LANG_MAP[tag.lower()]
        elif tag in ("withitn", "woitn"):
            itn = tag == "withitn"

    return {
        "text": clean,
        "language": language,
        "emotion": emotion,
        "event": event,
        "itn": itn,
    }


def _split_tagged_text(raw_text, total_duration):
    """Split SenseVoice output by embedded tag switches.

    SenseVoice embeds multiple tag sets when emotion/language changes
    mid-utterance: <|zh|><|NEUTRAL|>part1<|zh|><|SAD|>part2

    Returns list of (text, parsed_tags, proportional_start, proportional_end).
    """
    # Find all tag groups and their positions
    tag_pattern = re.compile(r'(<\|[^|]+\|>)+')
    matches = list(tag_pattern.finditer(raw_text))

    if not matches:
        return [(raw_text.strip(), _parse_tags(raw_text), 0, total_duration)]

    segments = []
    # Extract clean text between tag groups
    parts = []
    last_end = 0
    for i, m in enumerate(matches):
        if m.start() > last_end:
            # Text before this tag group belongs to previous segment
            pass
        tag_text = m.group()
        text_start = m.end()
        if i + 1 < len(matches):
            text_end = matches[i + 1].start()
        else:
            text_end = len(raw_text)
        clean = raw_text[text_start:text_end].strip()
        if clean:
            parsed = _parse_tags(tag_text + clean)
            parts.append((clean, parsed))

    if not parts:
        parsed = _parse_tags(raw_text)
        return [(parsed["text"], parsed, 0, total_duration)]

    # Assign proportional timestamps based on character count
    total_chars = sum(len(p[0]) for p in parts)
    pos = 0.0
    for text, parsed in parts:
        frac = len(text) / total_chars if total_chars > 0 else 0
        start = round(pos * total_duration, 3)
        pos += frac
        end = round(pos * total_duration, 3)
        segments.append((text, parsed, start, end))

    return segments


def transcribe(audio_path, language="auto", use_itn=True, use_vad=True):
    """Transcribe audio using SenseVoice.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    language : str
        Force language (zh/en/ja/ko/yue/auto). Default auto.
    use_itn : bool
        Apply inverse text normalization (number/date formatting).
    use_vad : bool
        Use VAD for segmentation with timestamps.

    Returns
    -------
    dict matching lyrics_whisper format:
        {segments: [{text, start, end}], language, emotion, event}
    """
    from funasr import AutoModel
    import librosa
    import soundfile as sf
    import tempfile
    import os

    kwargs = dict(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        disable_update=True,
    )
    if use_vad:
        kwargs["vad_model"] = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

    print("Loading SenseVoice model...")
    model = AutoModel(**kwargs)

    # SenseVoice expects 16kHz wav
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_wav.name, y, 16000)

    print(f"Transcribing with SenseVoice (lang={language})...")
    results = model.generate(
        input=tmp_wav.name,
        cache={},
        language=language,
        use_itn=use_itn,
        batch_size_s=60,
    )
    os.unlink(tmp_wav.name)

    import librosa as _lib
    total_duration = _lib.get_duration(filename=audio_path)

    segments = []
    emotions = []
    events = []
    overall_lang = "auto"

    for r in results:
        raw = r.get("text", "")
        if not raw:
            continue

        # Split by embedded tag switches, assign proportional timestamps
        parts = _split_tagged_text(raw, total_duration)
        for text, parsed, start, end in parts:
            if not text:
                continue
            segments.append({
                "text": text,
                "start": start,
                "end": end,
            })
            if parsed["emotion"]:
                emotions.append(parsed["emotion"])
            if parsed["event"]:
                events.append(parsed["event"])
            if parsed["language"] != "auto":
                overall_lang = parsed["language"]

    # Pick dominant emotion
    from collections import Counter
    overall_emotion = Counter(emotions).most_common(1)[0][0] if emotions else None
    overall_event = Counter(events).most_common(1)[0][0] if events else None

    # Clean up emotion/event if all segments have the same
    result = {
        "segments": segments,
        "language": overall_lang,
        "source": "sensevoice",
    }
    if overall_emotion:
        result["emotion"] = overall_emotion
    if overall_event:
        result["audio_event"] = overall_event

    print(f"SenseVoice: {len(segments)} segments, lang={overall_lang}"
          f"{f', emotion={overall_emotion}' if overall_emotion else ''}")

    return result
