# wave1: category-vote split 239 条四成员池重投（2026-09-27）

> 隔离台账 `category-vote` gate 239 条 split 行的人工待办消解执行记录。
> 主会话合并用；不改 tasks/C-execution.md（并发冲突，台账纪律记录落本文件）。

## 背景与动机

c_vote.py 首跑（batch 2026-09-25-catvote）时 doubao 未开通，实投三成员
（qwen-flash / qwen35b-local / glm-flash），872 行 agree=false 中 239 行 split
落隔离台账 priority=high 待人工。现 doubao-seed-2-1-pro-260915 已回池
（四成员 = qwen-flash/qwen38-local/doubao-vl 三强 + glm-flash 仲裁），
按「预标注优先：人不做机器能做的事」重投消解。

## 顺带修复：glm-flash 恒思考截断（本次消解率的关键杠杆）

- **现象**：首跑 glm parse_error 573/872（66%）；重投冒烟 3/3 parse_error。
- **根因**：glm-5.3-flash 恒思考（API 400 实测：不支持关闭，`thinking.type=low`
  同样被拒）；默认 `max_tokens=120` 被 reasoning 吃光（实测 reasoning ~131 tok），
  content 截断在 `{"verdict":"...` 半截（finish=length）。
- **修复**（`src/jxl/bin/vlm_pool.py`）：`call_role_vote` 的 max_tokens 支持
  per-spec 覆盖（默认 120 不变，其他消费方无感）；glm-flash spec 配
  `"max_tokens": 768`。reasoning 走独立 `reasoning_content` 字段，不污染
  `parse_verdict`。修复后 parse_error 降至 9/239（3.8%）。
- 冒烟对照：同 3 行，修复前 split 3/3 → 修复后 trusted 1 + arbited 1 + split 1。

## 代码改动（最小参数化）

1. `projects/s4det/bin/c_vote.py`：新增 `--keys-from`（只投指定行键文件，
   每行一个 `event|rid|object_id`）；键不在 agree=false 行中即 BadParameter
   （fail fast，No Silent Degradation）。`--out` 原生支持自定 sidecar。
2. `src/jxl/bin/vlm_pool.py`：glm max_tokens 修复（见上节）。

## 执行

```
python projects/s4det/bin/c_vote.py \
  --keys-from /mnt/data/s4det/preannot/votes_agree_false_rerun4.keys \
  --out       /mnt/data/s4det/preannot/votes_agree_false_rerun4.jsonl
# 池 = 默认四成员 qwen-flash,qwen38-local,doubao-vl,glm-flash；concurrency 6
# 239 行，16:22:07 → ~16:33，约 11 分钟，956 次模型调用
```

- 键文件来源：台账 239 条 `gate=category-vote` 且无 resolution 行的 id 去前缀。
- quarantine 追加为 no-op：`append_quarantine` 按 id 去重，重投 split 的 id
  （`2026-09-25-catvote:<key>`）已存在 → 台账行数 819 不变，消解证据由回写
  脚本以 resolution 落账。

## 结果

| 项 | 数字 |
|---|---|
| 重投行数 | 239（= 台账 239，键集合严格一致） |
| **消解（auto = trusted+arbited）** | **154（64.4%）**：trusted 97 + arbited 57 |
| 仍 split（待人工） | 85（35.6%），priority 仍 high 不动 |
| abstain / img_error | 0 |

auto verdict 逐类 top：garbage_bag 43、tarp_covered 23、carton 12、debris_misc 11、
construction_fence 10、garbage_pile 7、water_puddle 7、piled_goods 6、stall_cart 4、
barrier_fence 4、tricycle 4、construction_zone 3、foam_box 3、trash_can 3、
uncertain 2、其余 15 类各 1-2。

仍 split 85 的 plurality 分布：tarp_covered 11、carton 10、garbage_pile 9、
garbage_bag 8、piled_goods 6、debris_misc 5、foam_box 4、tricycle 4、uncertain 4、
person 3、oil_stain 3、barrier_fence 3、其余各 1-2——**集中在细粒度垃圾族互混
（tarp/carton/pile/bag/piled_goods/foam），正是人审价值最高的部分**。

逐模型票面健康（239 行）：qwen-flash ok 233 + api_error 6；qwen38-local ok 223 +
uncertain 16；doubao-vl ok 229 + uncertain 9 + parse_error 1；glm-flash ok 224 +
uncertain 6 + parse_error 9。

## 台账回写（j-quarantine-ledger：追加 resolution，不删旧行）

- 脚本：一次性数据迁移（逐行读、目标行追加字段、全量重写一次），
  备份 `/mnt/data/s4det/quarantine/quarantine-ledger.jsonl.bak-20260927`，
  原子替换（tmp + os.replace）。
- 字段格式（自包含字符串，沿台账既有 resolution 先例；不新增 `batch` 字段
  以免覆写原 batch=2026-09-25-catvote 史料）：
  - auto：`re-vote-4pool:auto: <trusted|arbited> verdict=<v> votes={...} (batch 2026-09-27-cvote-rerun, sidecar preannot/votes_agree_false_rerun4.jsonl)`
  - 仍 split：`re-vote-4pool:still-split: votes={...} (batch ..., sidecar ...)`

## 验收证据

- 台账 239 行全部有 resolution：auto 154 + still-split 85，other 0。
- 改动行 = 239，且每行仅新增 resolution 键，原字段（含旧 verdict/batch/priority）
  与备份逐字段一致；其余 580 行与备份完全一致；总行数 819 → 819。
- 旧 sidecar `/mnt/data/s4det/preannot/votes_agree_false.jsonl` 未动
  （mtime 2026-09-25 22:53:01；872 行，分布 trusted 607 / split 239 / arbited 26）。
- 新 sidecar `/mnt/data/s4det/preannot/votes_agree_false_rerun4.jsonl` 239 行，
  键集合 = 台账 239 键。
- ruff check 通过（c_vote.py、vlm_pool.py）；注意 c_vote.py 整文件在 HEAD
  即不通过 `ruff format --check`（存量紧凑风格），本次未整文件重排。

## 遗留

1. **85 行 still-split 待人工**（priority=high 未动，证据已更新为四成员票面）。
   建议人审聚焦细粒度垃圾族互混；可用 c_humanpack 流装配审核包。
2. **2 行 auto=uncertain**（qwen38+glm 一致判「目标不可辨」）：按共识语义自动
   定类为 uncertain——批 2 装配回填类别先验时该 2 行按 uncertain 处理，非叶类。
3. qwen-flash 6 次 api_error 未影响定类（其余成员足够票数）；glm 残余 9 次
   parse_error 同理。若后续要进一步压残量，可对这 15 行补票（成本极低）。
4. keys 文件 `/mnt/data/s4det/preannot/votes_agree_false_rerun4.keys` 与运行日志
   `votes_agree_false_rerun4.log` 留档为证据。

## 给主会话合并进 docs/EVAL-BENCHMARKS.md 的行文建议（只建议不改正文）

> **category-vote 门（agree=false 类别投票）**：参照物 =
> /mnt/data/s4det/preannot/votes_agree_false_rerun4.jsonl（四成员池 2026-09-27
> 重投票据，239 行）+ 台账 resolution 回写；级 ⚠️ 代理级（多模型投票共识，
> 非人工裁决）。基准数字：239 split 行四成员重投消解 64.4%（trusted 97 /
> arbited 57），残量 85 仍 split 待人工；glm-flash 恒思考截断修复（max_tokens
> 120→768）是消解率前提，parse_error 66%→3.8%。人工裁决级真值待 85 行
> still-split 人审后回填升级。
