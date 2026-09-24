# s4det 两级类目表（taxonomy）— 草案骨架 v0（2026-09-25）

> **状态：草案——未冻结，禁止作为标注依据**。冻结流程：任务书 A 聚类网格 → s4 会话看图命名/合并/剪枝（词典同步转正）→ 用户审批 → 状态改「v1 冻结」。
> 叶类 = 检测器标签（EN 短名）；粗类 = 事件域消费语义。**映射锚点两族并存**（盘点 §8.2）：① EventType 码（大多数叶类）；② llm_entries 类别串（ET15/16/19 在 event.json 无事件级存在——charging_bike 锚 "违规充电"、shop_display 锚 "占道经营"、scattered_garbage_zone 锚 "散落垃圾"）。细分硬条件：视觉可辨 ∧ 样本 ≥50-100；纯语义不可视区分不拆（红线）。

## vehicle（→ ET1/2/12、scene202 违停）

| 叶类 | 粗类映射 | 样本预期 | 状态 |
|---|---|---|---|
| ebike | ET2/EmergencyType2 | 大（13.6k 事件源） | 草案 |
| bicycle | ET2 | 中 | 草案（与 shared_bike 边界待聚类） |
| shared_bike | ET2 | 中 | 草案 |
| tricycle | ET2/ET12 | 中 | 草案 |
| car | ET1 | 小（467） | 草案 |
| truck | ET1 | 极小 | 草案（可能并回 car） |
| motorbike | ET2 | 中 | 草案（与 ebike 边界待聚类） |

## garbage（→ ET3/11/19、scene203）

| 叶类 | 粗类映射 | 样本预期 | 状态 |
|---|---|---|---|
| garbage_bag | ET3/ET19 | 大（12k 事件源；203 词频 5505） | 草案 |
| foam_box | ET3/ET11 | 大 | 草案 |
| carton | ET3/ET11 | 大 | 草案 |
| furniture_fallback | ET3 | 中 | 草案 |
| tarp_pile | ET11/ET3 | 大（词频 537） | 草案（命名待聚类定性） |
| plastic_loose | ET11 | 中 | 草案 |
| leaves_pile | ET3 | 小 | 草案 |
| dead_animal | ET3 | 极小 | 草案（<50 样本则并回+隔离台账） |
| garbage_pile | ET3/ET11/ET19 | 大 | 草案（aggregate 类，规范 §6） |
| debris_misc | ET3 兜底 | 中 | 草案（聚类后可能拆散并回） |

## vendor/occupy（→ ET6/7/12/16）

| 叶类 | 粗类映射 | 样本预期 | 状态 |
|---|---|---|---|
| stall_cart | ET12 | 中（1.1k） | 草案 |
| vendor_booth | ET12 | 中 | 草案（与 stall_cart 边界待聚类） |
| shop_display | ET16/ET7 | 中 | 草案 |
| construction_fence | ET6 | 小（64 事件，预期不够则与 scaffolding 并组观察） | 草案 |
| scaffolding | ET6/ET7 | 极小 | 草案 |
| piled_goods | ET11 | 中（497） | 草案 |

## facility（→ road 部件 + scene202；第二批数据，road VLM 打框时启用）

trash_can / fire_hydrant / bus_station / phone_booth / barrier_fence / manhole_cover / road_mirror / mailbox / power_station / sculpture / billboard —— 样本预期见盘点 §3.1 road 分布（trash_can 2978 大 / billboard 92 极小）。状态：草案（第二批）。

## phenomenon（→ ET13/14/18/19、road3）

water_puddle（3300 大）/ oil_stain（12 极小，观察类）/ road_damage（83 小）/ scattered_garbage_zone / construction_zone。状态：草案。

## state-carrier（→ ET15）

charging_bike —— ET15 仅 1 事件，**观察类**（数据极缺，词典先收形态，样本到达前不入训练）。状态：草案（观察）。

## 明确不入 v1

ET4（=road 部件类覆盖）/ ET5/8/9（店招=shop 域 sign.pt 检测线覆盖）/ ET10（无封闭类集，判定层兜底）——口径见 design §0。

---
## 修订日志

- v0（2026-09-25）：桌面草案骨架（自 research §7 细化 + 203 词频/TEXT 佐证），待聚类网格看图修订。
