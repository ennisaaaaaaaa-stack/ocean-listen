"""NetEase Cloud Music lyrics integration.
Adapted from Tinggu (SeithAsync, MIT).
"""
import json
import os
import pathlib
import re
import sys
import urllib.parse
import urllib.request

SEARCH_URL = "https://music.163.com/api/search/get"
LYRIC_URL = "https://music.163.com/api/song/lyric?id={id}&lv=1&tv=-1"
HEADERS = {"Referer": "https://music.163.com", "User-Agent": "Mozilla/5.0 (OceanListen/1.0)"}
TIMEOUT_S = 20
SEARCH_LIMIT = 10
DURATION_TOLERANCE_S = 2.0
TAIL_MISMATCH_S = 30.0
TIMESTAMP_RE = re.compile(r"\[(\d{1,3}):([0-5]?\d)(?:[.:](\d{1,3}))?\]")


class LyricError(Exception):
    pass


def _mmss(seconds):
    seconds = round(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _fetch_json(url, data=None):
    request = urllib.request.Request(url, data=data, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except OSError as error:
        raise LyricError(f"Cannot connect to NetEase: {getattr(error, 'reason', error)}") from None
    except (json.JSONDecodeError, ValueError):
        raise LyricError("NetEase returned invalid JSON") from None


def _search(query):
    body = urllib.parse.urlencode({"s": query, "type": 1, "limit": SEARCH_LIMIT, "offset": 0}).encode()
    result = _fetch_json(SEARCH_URL, data=body)
    try:
        songs = ((result.get("result") or {}).get("songs")) or []
        candidates = []
        for song in songs:
            artists = "/".join(artist.get("name", "") for artist in song.get("artists") or [])
            candidates.append({"id": song.get("id"), "name": song.get("name", ""),
                               "artist": artists, "duration_ms": int(song.get("duration") or 0)})
    except (AttributeError, TypeError, KeyError, ValueError):
        raise LyricError("NetEase response structure changed") from None
    return candidates


def _lyric(song_id):
    result = _fetch_json(LYRIC_URL.format(id=song_id))
    try:
        lrc = ((result.get("lrc") or {}).get("lyric")) or ""
        tlrc = ((result.get("tlyric") or {}).get("lyric")) or ""
    except (AttributeError, TypeError, KeyError):
        raise LyricError("NetEase response structure changed") from None
    return lrc, tlrc


def _extract_id_from_url(url):
    match = re.search(r"[?&#/]id=(\d+)", url) or re.search(r"song[/?#]+.*?id=(\d+)", url)
    if not match:
        match = re.search(r"id=(\d+)", url)
    if not match:
        raise LyricError("Cannot extract song ID from URL")
    return match.group(1)


def parse_lrc(text):
    lines = []
    for raw in text.splitlines():
        stamps = list(TIMESTAMP_RE.finditer(raw))
        if not stamps:
            continue
        content = TIMESTAMP_RE.sub("", raw).strip()
        if not content:
            continue
        for stamp in stamps:
            minutes = int(stamp.group(1))
            seconds = int(stamp.group(2))
            fraction = stamp.group(3) or "0"
            millis = int(fraction.ljust(3, "0")[:3])
            lines.append([round(minutes * 60 + seconds + millis / 1000, 3), content])
    lines.sort(key=lambda item: item[0])
    return lines


def _local_duration_s(audio_path):
    import librosa
    return float(librosa.get_duration(path=str(audio_path)))


def obtain_lyric(value, audio_path=None):
    """Get lyrics from NetEase by ID, URL, search query, or local file.
    
    Returns dict with: source, lrc, tlrc, lines.
    """
    path = pathlib.Path(value).expanduser()
    
    # Local file
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".lrc":
            lines = parse_lrc(text)
            if not lines:
                raise LyricError(f"No timestamped lyrics found in {path.name}")
            return {"source": f"file:{path.name}", "lrc": text, "tlrc": "", "lines": lines}
        if not text.strip():
            raise LyricError(f"{path.name} is empty")
        return {"source": f"file:{path.name}", "lrc": text, "tlrc": "", "lines": []}
    
    # URL
    if "http" in value:
        song_id = _extract_id_from_url(value)
        lrc, tlrc = _lyric(song_id)
        lines = parse_lrc(lrc)
        if not lrc.strip():
            raise LyricError("No lyrics available for this song")
        return {"source": f"netease:{song_id}", "lrc": lrc, "tlrc": tlrc, "lines": lines}
    
    # Numeric ID
    if value.isdigit():
        song_id = value
        lrc, tlrc = _lyric(song_id)
        lines = parse_lrc(lrc)
        if not lrc.strip():
            raise LyricError("No lyrics available for this song")
        return {"source": f"netease:{song_id}", "lrc": lrc, "tlrc": tlrc, "lines": lines}
    
    # Search query
    candidates = _search(value)
    if not candidates:
        raise LyricError(f"No results for '{value}'")
    
    if audio_path:
        duration = _local_duration_s(audio_path)
        hits = [c for c in candidates if abs(c["duration_ms"] / 1000 - duration) <= DURATION_TOLERANCE_S]
    else:
        hits = candidates[:1]
    
    if len(hits) >= 1:
        chosen = hits[0]
        lrc, tlrc = _lyric(chosen["id"])
        lines = parse_lrc(lrc)
        return {"source": f"netease:{chosen['id']}", "lrc": lrc, "tlrc": tlrc, "lines": lines}
    
    print(f"No candidate matched audio duration ({_mmss(duration)}). Top 5:", file=sys.stderr)
    for c in candidates[:5]:
        print(f"  id {c['id']} | {c['name']} | {c['artist']} | {_mmss(c['duration_ms'] / 1000)}", file=sys.stderr)
    raise LyricError("No duration match found, use --lyric-id to specify")
