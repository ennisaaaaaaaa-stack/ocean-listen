Ocean Listen 听海 — 合并方案
========================================

把 whale-listen-lyrics（MIDI音符 + whisper歌词）和 Tinggu 听骨（乐器识别 + 拆轨 + 嗓音质地 + 能量分析）合并为一个项目。


项目身份
--------

名字：Ocean Listen / 听海
寓意：鲸鱼在海里听——whale-listen 的延续，加上"听"的能力更完整。
协议：MIT
血统：fork 自 whale-listen（migratorywhale），整合 Tinggu（SeithAsync）的浅听/深听模块。
致谢链：whale-listen → whale-listen-lyrics → Ocean Listen；Tinggu → eryu → Ocean Listen。


三层听档设计
-----------

  浅听（Shallow）—— 快，轻依赖，约35秒/3分钟歌
    ├── BPM、调性、六段能量曲线、亮度趋势
    ├── 频段进场检测（低频/中低/中频/高频/气声）
    ├── 人声段落检测
    ├── PANNs 乐器识别（吉他/贝斯/鼓/钢琴/合成器/弦乐/管乐/风琴）
    ├── basic-pitch MIDI 音符提取（音高、力度、时长）
    └── 频谱图 PNG（Mel + Chroma + RMS + Band 四联图）

  深听（Deep）—— 慢，重依赖（torch + demucs + panns），约3-5分钟/3分钟歌
    ├── Demucs 六轨分离（人声/鼓/贝斯/吉他/钢琴/其他）
    ├── 每轨能量时间轴（乐器精确起止）
    ├── 嗓音质地（气息占比、空气感、混响尾巴、轻唱/爆发响度比）
    └── 每轨独立 MIDI 提取 ★ 新增——两个项目都没有的能力

  歌词（Lyrics）—— 可选，双来源
    ├── whisper 本地听写（离线，faster-whisper，支持多语言）
    ├── 网易云 API（精确标准歌词，±2秒时长护栏防错认）
    ├── 本地 .lrc / .txt 文件
    └── 时间轴对齐（歌词 ↔ 音符 ↔ 乐器）


★ 核心创新：每轨 MIDI 提取
---------------------------

两个项目单独都没有做到的事：

  原 whale-listen：整首歌跑 basic-pitch → 1068 个音符，但不知道哪个是贝斯哪个是人声
  原 Tinggu：Demucs 拆成六轨 → 知道乐器起止，但没有音符级数据

  Ocean Listen：Demucs 拆轨 → 每轨单独跑 basic-pitch
  结果："鼓 200 音符，贝斯 300 音符，人声 400 音符，吉他 168 音符"
  每个音符有音高、力度、时长，并且知道它属于哪件乐器。

  对编舞的意义：不只"这里没有鼓点"，而是"这里鼓停了，贝斯在走，
  人声留白——一个 2 秒的呼吸空间，身体该用 wave 填"。


文件结构
--------

  ocean-listen/
  ├── README.md           项目说明 + 双语 + 给AI的话
  ├── LICENSE             MIT，whale-listen 版权声明
  ├── NOTICES             THIRD_PARTY 致谢（whale-listen / Tinggu / eryu）
  ├── requirements.txt    浅听依赖（librosa, basic-pitch, faster-whisper）
  ├── requirements-deep.txt  深听依赖（torch, demucs, panns-inference）
  │
  ├── ocean.py            主入口 CLI（三档 + 歌词参数）
  │
  ├── modules/
  │   ├── notes.py        basic-pitch MIDI 提取（← whale-listen）
  │   ├── lyrics_whisper.py  本地 whisper 听写（← whale-listen）
  │   ├── lyrics_netease.py  网易云歌词 API（← Tinggu lyrics.py）
  │   ├── structure.py    BPM/调性/能量/频段/亮度（← Tinggu analyze_song.py）
  │   ├── instruments.py  PANNs 乐器识别（← Tinggu analyze_song.py）
  │   ├── stems.py        Demucs 拆轨 + 每轨时间轴（← Tinggu ears.py）
  │   ├── voice.py        嗓音质地分析（← Tinggu ears.py）
  │   ├── per_stem_notes.py  每轨 MIDI 提取 ★ 新代码
  │   ├── visualize.py    频谱图生成（← Tinggu analyze_song.py）
  │   └── report.py       统一报告输出（文本 + JSON）
  │
  ├── whale_listen.py     原版 whale-listen 保留（向后兼容）
  └── music/              音乐库


CLI 设计
--------

  # 浅听（快速结构 + 音符 + 乐器标签）
  python ocean.py song.mp3

  # 深听（+ 拆轨 + 嗓音质地 + 每轨音符）
  python ocean.py song.mp3 --deep

  # 歌词 - whisper 本地听写
  python ocean.py song.mp3 --lyric whisper --language zh

  # 歌词 - 网易云
  python ocean.py song.mp3 --lyric "歌曲名 歌手"

  # 歌词 - 本地文件
  python ocean.py song.mp3 --lyric ./song.lrc

  # 全量（深听 + whisper歌词）
  python ocean.py song.mp3 --deep --lyric whisper --language en

  # 强制重算（忽略缓存）
  python ocean.py song.mp3 --deep --force


输出格式
--------

  JSON 包含：
  {
    "name": "soaked",
    "duration": 176.6,
    "bpm": 96,
    "key": "F#",
    "segments": [...],          // 六段能量
    "brightnessTrend": "falling",
    "instruments": {...},       // PANNs 识别结果
    "notes": [...],             // 全曲 MIDI 音符（浅听）
    "stemTimeline": {...},      // 六轨乐器起止（深听）
    "stemNotes": {...},         // 每轨 MIDI 音符（深听）★ 新
    "voiceProfile": {...},      // 嗓音质地（深听）
    "lyrics": [...],            // 时间轴歌词
    "timeline": [...]           // 歌词 + 音符 + 乐器三维对齐
  }

  文本报告：继承 Tinggu 的人类可读格式，
  在歌词段落附上音符密度和乐器状态。
  频谱图 PNG：Tinggu 原版四联图。


依赖与内存
----------

  浅听：librosa + basic-pitch + faster-whisper（已验证，约2.2GB内存峰值）
  深听：+ torch + demucs + panns（约2.5GB额外磁盘，运行时峰值待测）
  串行执行避免内存叠加：浅听 → 拆轨 → 每轨分析 → 歌词（各阶段独立）


致谢与沟通
----------

  NOTICES 文件内容：

  This project incorporates code from:
  1. whale-listen by migratorywhale (MIT)
     - MIDI note extraction via basic-pitch
     - Whisper lyrics transcription
  2. Tinggu 听骨 by SeithAsync (MIT)
     - Shallow/deep analysis architecture
     - PANNs instrument recognition
     - Demucs stem separation
     - Voice profile analysis
     - NetEase lyrics integration
     Which itself incorporates code from:
     - eryu by sebastianevan200-stack (MIT)

  tiexin去找 SeithAsync 说的时候，可以说：
  "我把你的听骨和另一个项目 whale-listen 合并了，
  叫听海 Ocean Listen。你的浅听深听架构完整保留了，
  致谢在 NOTICES 里。加了一个新功能——Demucs 拆完轨之后
  每轨单独跑 MIDI 提取，这样不只是知道乐器什么时候在响，
  还知道它响了哪些音。"


工作顺序
--------

  Phase 1: 重命名仓库 + 写 NOTICES + 更新 README 框架
  Phase 2: 整合 Tinggu 浅听模块（structure + instruments + visualize）
  Phase 3: 整合深听模块（stems + voice）
  Phase 4: 新增 per_stem_notes（每轨 MIDI 提取）
  Phase 5: 整合网易云歌词模块
  Phase 6: 统一 CLI + 报告输出
  Phase 7: 用 Soaked 做端到端测试（有编舞记录做对照）
