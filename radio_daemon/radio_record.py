#!/usr/bin/env python3
"""
radio_record.py — Ocean Listen 电台录制（短命进程，systemd timer 拉起）
zhaozhao 2026-08-29。每段独立进程：死流只损失一段，分段即重试。

用法：
  python3 radio_record.py                      # 按 sources.json 全部源录 30min
  python3 radio_record.py --source dronezone   # 只录指定源
  python3 radio_record.py --duration 60        # 测试：只录 60s
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

BASE = pathlib.Path(__file__).parent
SOURCES_FILE = BASE / "sources.json"
RAW_DIR = BASE / "radio_raw"

# 录制窗口（本地时间）。窗口外不录。
START_HOUR, END_HOUR = 23, 7
SEGMENT_SECONDS = 1800  # 30 min


def in_window(now=None):
    """23:00-07:00 跨午夜窗口。"""
    h = (now or time.localtime()).tm_hour
    if START_HOUR > END_HOUR:  # 跨午夜
        return h >= START_HOUR or h < END_HOUR
    return START_HOUR <= h < END_HOUR


def record_one(source: dict, duration: int) -> bool:
    name = source["name"]
    url = source["url"]
    ts = time.strftime("%Y%m%d-%H%M")
    day_dir = RAW_DIR / time.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    out = day_dir / f"{name}-{ts}.mp3"

    # .part 写完再改名：分析器只见完整段，崩溃不留半截文件
    part = out.with_suffix(".part.mp3")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", url, "-t", str(duration), "-vn", "-acodec", "copy",
        str(part),
    ]
    print(f"[record] {name}: {url} -> {out.name} ({duration}s)")
    t0 = time.time()
    r = subprocess.run(cmd, timeout=duration + 120)
    if r.returncode != 0 or not part.exists() or part.stat().st_size < 100_000:
        print(f"[record] {name}: FAILED rc={r.returncode}, cleaning part file")
        part.unlink(missing_ok=True)
        return False
    part.rename(out)
    print(f"[record] {name}: OK {out.stat().st_size/1e6:.1f}MB in {time.time()-t0:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="only this source name")
    ap.add_argument("--duration", type=int, default=SEGMENT_SECONDS)
    ap.add_argument("--force", action="store_true", help="ignore recording window")
    args = ap.parse_args()

    if not args.force and not in_window():
        print(f"[record] outside window {START_HOUR}:00-{END_HOUR}:00, skip")
        return

    sources = json.loads(SOURCES_FILE.read_text())
    ok = 0
    for s in sources:
        if args.source and s["name"] != args.source:
            continue
        try:
            ok += record_one(s, args.duration)
        except subprocess.TimeoutExpired:
            print(f"[record] {s['name']}: TIMEOUT")
    print(f"[record] done: {ok} ok")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
