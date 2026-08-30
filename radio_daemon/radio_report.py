#!/usr/bin/env python3
"""
radio_report.py — Ocean Listen 电台收听报告（每天早上 systemd timer 拉起）
zhaozhao 2026-08-29。汇总昨日：录了多少、粗筛扔了什么、策展候选一览。
报告写 radio_reports/（纯文件 v1——新表要hui点头，先用文件，量小够用）。
"""
import datetime as dt
import json
import pathlib
import sqlite3

BASE = pathlib.Path(__file__).parent
DB_PATH = BASE.parent / "ocean_music_box.db"
RAW_DIR = BASE / "radio_raw"
DONE_DIR = BASE / "radio_done"
REPORTS_DIR = BASE / "radio_reports"


def main():
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    lines = []
    lines.append(f"♪ Ocean Listen 电台日报 — {yesterday} 录制（{today} 出报）")
    lines.append("")

    # 1) 昨日录制的段
    day_dir = RAW_DIR / str(yesterday)
    done_files = list(DONE_DIR.glob(f"*-{yesterday.strftime('%Y%m%d')}*.mp3"))
    raw_files = list(day_dir.glob("*.mp3")) if day_dir.exists() else []
    total_mb = sum(f.stat().st_size for f in raw_files + done_files) / 1e6
    lines.append(f"录制：{len(raw_files) + len(done_files)} 段 / {total_mb:.0f}MB")

    # 2) 入库的（songs 表里昨日新增的 radio-）——DB 不在也能出报告
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT name, bpm, key_name, duration FROM songs "
            "WHERE name LIKE 'radio-%' AND date(created_at) = ?",
            (str(yesterday),),
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"[report] DB unavailable: {e}")
        rows = []
    if rows:
        lines.append("")
        lines.append(f"值得听（已入库，待zhaozhao审）：{len(rows)} 段")
        for name, bpm, key, dur in rows:
            lines.append(f"  · {name} | {bpm}BPM {key} | {dur/60:.0f}min")
    else:
        lines.append("值得听：0 段（昨晚要么全被粗筛，要么还没分析完）")

    # 3) 粗筛统计（done 目录里的 dropped 哪些）
    # v1.1：move_to_done 时把 reason 写进 sidecar json，报告才有的精确写。先看有没有
    lines.append("")
    lines.append("粗筛：见 radio_done/（v1 先人工看，统计行 v1.1 补）")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{yesterday}.md"
    out.write_text("\n".join(lines))
    print(f"[report] {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
