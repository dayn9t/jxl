# jxl 项目代办

- [x] [2026-09-25→09-25] 隔离台账待人工批量处理：s4det 579 条 — s4 会话代用户裁决全量「确认丢弃」（legacy 无 event.json 结构性缺失，盘点 §8.3），resolution 已回写台账，零逐条处理（台账 /mnt/data/s4det/quarantine/quarantine-ledger.jsonl）#s4det #quarantine
- [~] [2026-09-22] 隔离台账待人工批量处理：teller-flat 52 条（坐姿业务员可回收框，v7.1 前置）——已打包工作台 r02 轮，0/52 等裁（台账 /mnt/data/jiang/ws/sgcc/person/datasets/sgcc-n001/crop640_persons/gencheck/quarantine/2026-09-20-teller-flat.jsonl）#quarantine
- [x] [2026-09-27→09-27] t4_preannot 能力账落档：run 写 pool-snapshots + score 落 capability-ledger.jsonl（c734ecd，快照/台账双冒烟过）#s4det
- [ ] [2026-09-27] 终版合并切分必须消费评估排除清单 /mnt/data/s4det/eval-gt/exclusion-events.jsonl（377 帧：T4-b1 8 + T4-b2 369 与训练池重叠，防评估泄漏；合并工具加 --exclude 接线）#s4det
