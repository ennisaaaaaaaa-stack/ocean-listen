#!/usr/bin/env python3
"""
radio_analyze.py — Ocean Listen 电台段分析（短命进程，systemd timer 每 10min 拉起）
zhaozhao 2026-08-29。串行处理 radio_raw/ 未分析段；单次调用最多处理 N 段（防止跑过头撞上录像时段）。

流程 / 段：
  1. SenseVoice 滑窗贴标签（speech/music 粗判）——混合台路由用
  2. music 台：强制音乐管线（run_shallow，无视 classifier——ambient 会骗过它）
  3. 粗筛客观题：重复(embedding)/说话占比/静音 → 扔的进报告统计
  4. 过粗筛的段写八音盒 DB（songs+structure，触发器自动进 FTS）
"""
import argparse
import json
import pathlib
import sqlite3
import subprocess
import sys
import urllib.request

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent))  # ocean-listen 根目录（ocean.py, modules/）

RAW_DIR = BASE / "radio_raw"
DONE_DIR = BASE / "radio_done"          # 分析完的段挪这里，滚动清理只碰 radio_raw？不——清理按 mtime，done 也清
SOURCES_FILE = BASE / "sources.json"
DB_PATH = BASE.parent / "ocean_music_box.db"
EMBED_URL = "http://localhost:18001/embed_batch"
DUP_COSINE = 0.92
SPEECH_RATIO_DROP = 0.6
SILENCE_RMS = 0.01
MAX_BATCH = 4  # 单次最多处理段数（浅听 ~1GB 内存串行，防跑过头）


def load_sources() -> dict:
    return {s["name"]: s for s in json.loads(SOURCES_FILE.read_text())}


def source_of(path: pathlib.Path, sources: dict) -> dict:
    # 文件名 radio-<source>-YYYYMMDD-HHMM.mp3 / <source>-YYYYMMDD-HHMM.mp3
    return sources.get(path.name.split("-")[0])


def embed(texts: list) -> list | None:
    payload = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(EMBED_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["embeddings"]
    except Exception as e:
        print(f"[embed] error: {e}")
        return None


def sensevoice_tag(audio: str) -> dict:
    """滑窗贴标签：返回 {speech_ratio, tags: [{t, label}]}。一次推理，粗筛+路由共用。"""
    from funasr import AutoModel
    # 60s 窗口整段一次（v1 从简；窗口滑窗进 v1.1 如果混合台需要更细）
    model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)
    res = model.generate(input=audio, language="auto")
    text = res[0].get("text", "")
    # SenseVoice 输出形如 <|startlocal|><|en|><|NEUTRAL|><|Speech|><|woitn|>content
    tags = [t for t in text.split("<|") if t.endswith("|>") or t and False]
    is_speech = "<|Speech|>" in text
    is_music = "<|Music|>" in text or "<|BGM|>" in text
    return {"raw": text, "is_speech": is_speech, "is_music": is_music}


def analyze_segment(path: pathlib.Path, source: dict) -> dict:
    """单段全流程。返回 {action: keep|drop|error, reason, ...}"""
    name = f"radio-{path.stem}"

    # 1) SenseVoice 贴标签（30min 段太大，先切 3 个抽样点：头 60s / 中 60s / 尾 60s 抽one）
    # v1 从简：抽中段 60s 判类别；混合台细路由 v1.1
    probe = BASE / "tmp"
    probe.mkdir(exist_ok=True)
    probe_file = probe / path.name
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip() or 0)
    mid = max(0, dur / 2 - 30)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(mid), "-i", str(path), "-t", "60",
                    "-vn", "-acodec", "copy", str(probe_file)], check=True)
    sv = sensevoice_tag(str(probe_file))
    probe_file.unlink(missing_ok=True)

    # 音乐台但抽中段是 speech>60%？——v1 用整段单标签近似
    if source["type"] == "music" and sv["is_speech"] and not sv["is_music"]:
        return {"action": "drop", "reason": "talk_not_music", "sv": sv["raw"][:200]}

    # 2) 强制音乐管线
    import ocean
    cache_dir = BASE / "ocean_cache_radio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = ocean.run_shallow(str(path), cache_dir, force=False)
    data["name"] = name
    data["sourcePath"] = str(path)
    data["radioSource"] = source["name"]

    # 3) 静音粗筛（用 shallow 的 segments avgEnergy）
    segs = data.get("segments", [])
    if segs:
        silent = sum(1 for s in segs if s.get("avgEnergy", 0) < SILENCE_RMS) / len(segs)
        if silent > 0.8:
            return {"action": "drop", "reason": "silence"}

    return {"action": "keep", "data": data, "sv": sv["raw"][:200]}


def dup_check(data: dict, path: pathlib.Path, no_db: bool = False) -> bool:
    """重复播放粗筛：BPM+key+notes 三元组 vs 近 7 天已入库段。embedding 精筛 v1.1"""
    if no_db:
        return False
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name, bpm, key_name, total_notes FROM songs "
        "WHERE name LIKE 'radio-%' AND imported_at > datetime('now', '-7 days')",
    ).fetchall()
    conn.close()
    if not rows:
        return False
    # v1 从简：BPM+key+notes 三元组近似重播判定；embedding 精筛 v1.1
    for sid, nm, bpm, key, notes in rows:
        if bpm and data.get("bpm") and abs(bpm - data["bpm"]) <= 3 \
           and key == data.get("key") \
           and notes and data.get("total_notes") \
           and abs(notes - data["total_notes"]) / max(notes, 1) < 0.15:
            print(f"[dup] {path.name} ~= {nm} (bpm/key/notes)")
            return True
    return False


def import_to_box(data: dict) -> int:
    """按 import_to_music_box.py 的 INSERT 模式写库。FTS 触发器自动处理。"""
    draft = generate_draft(data)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO songs (name, source_path, analysis_json, bpm, key_name, duration,"
        " total_notes, brightness_trend, spectrogram_path) VALUES (?,?,?,?,?,?,?,?,?)",
        (data["name"], data.get("sourcePath", ""), json.dumps(data, ensure_ascii=False),
         data.get("bpm"), data.get("key"), data.get("duration"),
         data.get("total_notes"), data.get("brightnessTrend"), data.get("spectrogram", "")),
    )
    song_id = cur.lastrowid
    cur.execute("INSERT INTO structure (song_id, auto_draft) VALUES (?,?)",
                (song_id, draft))
    conn.commit()
    conn.close()
    return song_id


def generate_draft(data: dict) -> str:
    """照抄hui的 generate_structure_draft 精简版（电台段专用）。"""
    lines = []
    bpm, key, dur = data.get("bpm", "?"), data.get("key", "?"), data.get("duration", 0)
    lines.append(f"♪ 电台段 | {data.get('radioSource','?')} | BPM {bpm} | Key {key} | {dur/60:.0f}min")
    lines.append(f"♪ Notes: {data.get('total_notes','?')} | Brightness: {data.get('brightnessTrend','?')}")
    segs = data.get("segments", [])
    if segs:
        lines.append("--- Energy ---")
        for i, s in enumerate(segs):
            bar = "█" * int(s.get("avgEnergy", 0) * 50)
            lines.append(f"  Seg{i+1} avg={s.get('avgEnergy',0):.3f} {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=MAX_BATCH)
    ap.add_argument("--file", help="analyze one specific file (bypasses scan)")
    ap.add_argument("--no-db", action="store_true",
                    help="dry-run: skip dup_check and DB import (local dev)")
    args = ap.parse_args()

    sources = load_sources()

    if args.file:
        targets = [pathlib.Path(args.file)]
    else:
        targets = sorted(p for p in RAW_DIR.rglob("*.mp3"))

    processed = 0
    stats = {"keep": [], "drop": [], "error": []}
    for path in targets:
        if processed >= args.max:
            break
        src = source_of(path, sources)
        if not src:
            print(f"[skip] {path.name}: unknown source")
            continue
        try:
            r = analyze_segment(path, src)
        except Exception as e:
            stats["error"].append({"file": path.name, "err": str(e)[:200]})
            print(f"[error] {path.name}: {e}")
            continue
        if r["action"] == "drop":
            stats["drop"].append({"file": path.name, "reason": r["reason"]})
            move_to_done(path, "dropped")
        elif r["action"] == "keep":
            if dup_check(r["data"], path, no_db=args.no_db):
                stats["drop"].append({"file": path, "reason": "dup"})
                move_to_done(path, "dropped")
            elif args.no_db:
                print(f"[keep-dry] {path.name} -> would import (dry-run)")
                move_to_done(path, "kept")
            else:
                sid = import_to_box(r["data"])
                stats["keep"].append({"file": path.name, "song_id": sid})
                move_to_done(path, "kept")
        processed += 1

    print(json.dumps(stats, ensure_ascii=False, indent=1))


def move_to_done(path: pathlib.Path, why: str):
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    dest = DONE_DIR / path.name
    path.rename(dest)


if __name__ == "__main__":
    main()
