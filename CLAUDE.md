# jxl — 视觉域承接库（模型训练/数据处理集中仓）

> 定位（承接库消费纪律正本：`~/.claude/skills/j-polyglot-monorepo/SKILL.md`「承接库消费纪律」节）：
> iap/s4 等下游项目的模型训练、数据处理、标注构建集中于此；依赖单向（下游→本仓，本仓永不依赖下游）；
> 项目管理面在 `projects/{sgcc,s4det,shtm}/`（正本文档+台账），训练活动发生在本仓。

## 快速导航

- `projects/sgcc/` — n001 窗口 person 检测/属性分类/ReID（README 状态表+research 报告族）
- `projects/s4det/` — s4 事件域检测器（冻结中，`FREEZE-20261005.md`）
- `projects/shtm/` — 垃圾桶检测（子项目冻结）
- `src/jxl/bin/` — 正本工具位（vlm_pool/osnet_scratch 等，git 有史）；`src/jxl/vdt/` — ReID 推理
- `gencheck/` — 现场脚本镜像（正本在 src，数据盘 `~/ws/sgcc-mma/training/person/.../gencheck/`）
- `TODO.md`/`todos.md` — 任务队列正本；`.claude/skills/` — 仓内 skill（5 个）

## 关键纪律指针

- **bug 触发用例固化**：`/home/jiang/cc/cc/j-cc/skills/j-bug-case-golden/SKILL.md`
  （j-cc 仓，repo-only 挂指针形态）——凡暴露缺陷的真实输入（帧图/音频/数据文件）必须固化为
  自包含 ground truth 用例再关闭 bug；适用判别「触发输入是可保存的媒体/数据文件吗」
- **数据飞轮触发**：新数据到达必触发五件套盘点（见 `.claude/skills/data-flywheel/` + memory）
- **训练机**：n902（sgcc-n902，用户指定）；sgcc3/sgcc-0003 已转在役生产节点不训练；ds 驱动待修

## 环境

- Python 一律 `.venv/bin/python`（uv 管理）；依赖按 j-python-tech-selection 锁定表
- 数据盘：`/home/jiang/ws/sgcc-mma/`（16T；4T 旧盘接口故障已退役流程中）——gold 语料在 `gold/n001/`
