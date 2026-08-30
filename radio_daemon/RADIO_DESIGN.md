# Ocean Listen 电台守护 · 需求-设计文档
（zhaozhao起草 2026-08-29，hui需求纸 + tiexin三轮讨论收敛而成。权属：Ocean Listen 是zhaozhao的项目，电台链路zhaozhao拍板；hui=共建者+托管方+提名权。）

## 目标
后台抓电台流 → 异步分析 → 早上"昨晚听了什么"收听报告。八音盒策展照旧（策展人现在是zhaozhao，hui保留提名权）。

## 与hui需求纸的差异（已实测验证）
1. **FTS 不用手动写**：music_fts 有 4 个 AFTER INSERT 触发器（story/flow/structure/tags）自动同步。hui纸里"每层写完必须配 music_fts 行"是错的，她的 import_to_music_box.py 注释才对。daemon 不得画蛇添足手动插 FTS。
2. **ambient 骗过分类器（实测坐实）**：Drone Zone 60s 样本 → classifier 判 VOICE conf=0.8（perc=0.19<0.2，PANNs 把 ambient 铺底全认成人声，vocal_cov=100%）。电台段必须**源级强制管线**：音乐台强制 music/solo，不信自动分类。
3. **流不都是实时的**：SomaFM ice1 实测 1.5x 下流（服务器有预滚缓冲）。但设计仍按实时算：30min 段≈30min 录制时间。能超速是意外之喜不是承诺。

## 架构（全短命进程，无常驻 daemon）

```
录像 systemd-timer (23:00-07:00 每30min)
  → ffmpeg -i URL -t 1800 落盘 radio_raw/YYYY-MM-DD/HH-MM.mp3
     （死流只损失一段，分段即重试）
分析 systemd-timer (每10min看收件箱，白天为主)
  → queue_scan.py 串行处理 radio_raw 未分析段
     → SenseVoice 滑窗贴标签（speech/music 段级判定）
     → 音乐段强制音乐管线（run_shallow，无视 classifier）
     → 产物写八音盒 DB（结构触发器自动进 FTS）
     → 粗筛报告行（重复/说话/静音，扔的留统计）
清理 systemd-timer (每天一次)
  → 保留7天；DB 引用的音频路径豁免（不变量：有卡片不许没歌）
报告 cron/timer (每天早上)
  → 昨晚段数/时长/扔了什么/值得听的候选+一句话理由
```

## 源配置（源级，不是全局）
每台一份 source 配置（JSON）：
```json
{"name": "dronezone", "url": "https://ice1.somafm.com/dronezone-128-mp3", "type": "music", "pipeline": "solo"}
{"name": "resonance", "url": "...", "type": "mixed", "pipeline": "auto"}
```
- type=music → 整段强制音乐管线（防 ambient 骗分类器）
- type=mixed → SenseVoice 滑窗分拣：音乐子段走音乐，谈话子段走转写四层
- 转写四层（tiexin拍板方向）：层0滑窗标签（粗筛副产品）/ 层1全文转写默认全开（SenseVoice CPU=音频时长÷10，便宜到不用省）/ 层2 LLM 理解按需（看报告点播）/ 层3 说话人分离 v1 不做

## 粗筛（机器只做客观题）
自动扔（进报告统计行，不进策展队列）：
- 重复播放：BGE-M3 embedding 对比近 7 天段（cosine>0.92 → dup）
- DJ 说话/广告：SenseVoice 判 speech 占比 > 60% 且音乐台 → 整段扔
- 靍音：RMS < 0.01 占比 > 80% → 扔
其余全部进zhaozhao待审队列（v1 零自动入库，八音盒是策展馆不是垃圾桶）。

## 八音盒接口（schema 不动，按现有 INSERT 模式写）
- INSERT INTO songs → INSERT INTO structure(auto_draft) → 触发器自动进 FTS
- 段命名：`radio-{source}-{YYYYMMDD-HHMM}`（songs.name 唯一）
- embedding 走 VPS localhost:18001（embed-server.service，BGE-M3）
- 静音检测用 ocean.py 现有 RMS 数据（shallow JSON 里有 segments avgEnergy）

## 存储与调度
- 128kbps ≈ 1.3GB/天，radio_raw/ 滚动 7 天
- 分析调度窗口：白天 09:00-21:00 为主，避开hui夜间在线时段是反的——**避开白天hui醒着用机器的时段**，分析排夜里也是选项（机器闲）
- VPS 内存 7.5G 总/3.4G 可用：浅听单进程 ~1GB，串行跑，不并发
- 收听报告写 DB（新表 radio_reports，schema 变更须hui点头——schema 归她管）
```

## 挂流听（提前进 v1，tiexin拍板）
同一底座两种触发：daemon=定时触发，挂流=手动触发。
- 滚动缓冲 15min 内存环形录音，平时不落盘
- 触发截取：缓冲落盘+续录 30min → 进转写队列 → 八音盒条目"转写中"→ 转完可读
- agent 截的段进转写区不用审；想进策展区照样过策展（zhaozhao）的手
- 优先级：手动截取 > 报告深挖 > 夜间批处理（一条队列三个入口）

## 开放项（后续）
- 微信推送：不进 v1
- beats/downbeat：独立线，ambient 无节拍用不上
- Resonance FM 源验证：混合台路由 v1 先只配 SomaFM 两台，混合台等手上有真实样本再加
- LLM 理解层（层2）触发方式：报告里点播，具体交互等第一份报告出来再定
