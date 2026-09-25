# s4det 两级类目表（taxonomy）v1-暂定冻结（2026-09-25，聚类网格人审后）

> **状态：v1-暂定冻结**——依据用户循环指令「持续推动直到所有任务完成」的standing授权（2026-09-25 s4 会话裁定）；**用户审批权继续有效**：批准即转「v1 冻结」终态，有改动则批次间修正（类链机制改动面小，B② 已证）。任务书 C 第一批仅用 ebike/garbage_bag 两粗类锚最无争议类。
> 流程：任务书 A 聚类网格 → s4 会话看图命名/合并/剪枝（本版）→ 用户审批（暂定冻结下继续有效）→「v1 冻结」。
> 叶类 = 检测器标签（EN 短名）；粗类 = 事件域消费语义。**映射锚点两族并存**（盘点 §8.2）：① EventType 码；② llm_entries 类别串（ET15/16/19 无事件级存在）。
> 细分硬条件：视觉可辨 ∧ 样本 ≥50-100；纯语义不可视区分不拆（红线）。
> **三路径分派**（design §2/2b/2c）：本表登记全部类目池并标注执行体归属（附录 A 总表）——**D1 主检测器类目池 22**（已训 2 / 批 2 直入 12 / 挂验证 5 / 观察候补 3；含 person 辅助证据类，用户 2026-09-25 拍板）+ **D2 烟火 2** + **D3 哨兵 11** + **D4 VLM**（非类目）+ **证据链道具 8**（预研）。中文名权威=本表。
> **网格人审总发现**：vehicle 码 11 簇质量高；garbage 码 1/2/4 噪声率高（16 簇中约 9 簇为石球/栅栏/路缘/单车等误框）；码 5 实为「城市杂项」大杂烩（摊位/单车群/行人/工程车混框）——**平台框仅作位置先验、必须共识仲裁**（CODEBOOK 已更新）。

## vehicle（→ ET1/2、parking 判定）

| 叶类 | 粗类锚 | 图证 | 决定 | 备注 |
|---|---|---|---|---|
| ebike | ET2 + llm"非机动车乱停" | ✅c01 并排停放/c03/c08、c04+c15 骑乘中、c16 载货、c13 低质 | **保留（含 motorbike 并入）** | 摩托与电动外观难区分（红线不拆）；骑行/停放是状态不入检测类；成排停放每辆一框（规范 §7） |
| shared_bike | ET2 | ✅vehicle c09 美团黄、vendor c11 蓝/c12 青桔 | **保留** | 统一涂装视觉可辨 |
| bicycle | ET2 | 聚类未见独立簇 | 保留（待 P3 验证样本） | parking 判定语义需要 |
| tricycle | ET2/ET12 | ✅vendor c08 回收三轮（载废品+招牌） | **保留** | 载货斗/封闭棚形态 |
| car | ET1 | ET1 65 样本未成独立簇（k=16 被并） | **保留；truck 并回 car** | 面包车/货车为 car 内形态；判定层需卡车区分时 P5 再拆 |
| **person** | 辅助证据类（无粗类锚） | 批 2 起全量标注 | **批 2 新增（用户 2026-09-25 拍板）** | 违停状态判定数据源（rider_visible）+ 证据链一级锚点；COCO 底座最成熟类，零风险 |
| ~~motorbike~~ | — | — | **并入 ebike** | 视觉不可辨（红线） |
| ~~truck~~ | — | — | **并入 car** | 样本 <50 硬条件 |

## garbage（→ ET3/11、llm"丢弃垃圾/占道堆物/散落垃圾"）

| 叶类 | 粗类锚 | 图证 | 决定 | 备注 |
|---|---|---|---|---|
| garbage_bag | ET3 | ✅c01 吨袋/编织袋 n156、c04 白袋堆 n118 | **保留** | 黑/白/橙袋+编织袋+吨袋；最高频形态 |
| carton | ET3/ET11 | ✅c02 纸箱堆 n139、c05 柜台纸箱 n116 | **保留** | |
| foam_box | ET3 | ✅c06/c07 | **保留** | |
| garbage_pile | ET3/ET11 | ✅c03 混合不可数堆 n131 | **保留** | aggregate 类（规范 §6 口诀） |
| tarp_covered | 待定 | ⚠️c14 篷布覆盖**车**（TEXT 样本为覆盖**堆放物**） | **观察类→hardcase** | 覆盖物歧义（车 vs 堆），粗类锚不定 → 人工核实后定 |
| plastic_loose | ET11 | 203 词频 240 佐证，聚类未见 | 保留（待 P3） | |
| leaves_pile / dead_animal | ET3 | 未见 | **降为观察类** | 样本不足硬条件；dead_animal 预期极小 |
| debris_misc | 兜底 | 未见 | 保留兜底 | 聚类后仍散不归者入此 |

## vendor/occupy（→ ET6/7/12/16、llm"占道经营"）

| 叶类 | 粗类锚 | 图证 | 决定 | 备注 |
|---|---|---|---|---|
| construction_fence | ET6 | ✅c01 黄围挡「正在施工」n113 | **保留** | |
| scaffolding | ET6/ET7 | ✅c03 脚手架+防护网 n96 | **保留** | |
| vendor_booth | ET12 | ✅c02 棚/货架摊 n107 | **保留** | 固定/半固定结构摊位 |
| stall_cart | ET12 | ✅c04 遮阳伞+流动摊 n87 | **保留** | 伞摊/售卖车/地摊布 |
| piled_goods | ET11 | ✅c06 纸箱编织袋堆地 n67 | **保留** | 整齐度区分垃圾（misc.md） |
| shop_display | ET16（llm 锚"占道经营"） | 未见独立簇（补样 121 散） | 保留（待 P3 验证） | |

## facility（第二批 road 数据；2026-09-25 起引入路径分派——「固定目标哨兵路径」见 design §2b）

**路径分派总则**（用户口述决策，2026-09-25）：动态可移动、形态确定的目标 → 检测器；静止固定位置、形态/边界不确定的目标 → GPS+视频+VLM 哨兵路径（机制：GPS 缩范围 → 附近 ~5s@1fps 短视频 → VLM 定位/筛图/描述存档 → 前后状态对比；详见 research/2026-09-25-固定目标GPS-VLM路径提案.md）。

| 叶类 | 路径归属 | 裁定理由 |
|---|---|---|
| billboard | **哨兵路径**（检测器不适用） | 墙上区域、无边界、内容自由——bbox 语义失效；业务价值（褪色/涂鸦/脱落）本质是状态对比 |
| sculpture | **倾向哨兵路径**，批 2 样本验证检测器胜任性后定 | 题材开放、形态特征弱；按用户要求以观测样本判定（聚类独立簇 + 试标 50 框） |
| trash_can | 检测器 ✓ | 可移动（移位/失踪是业务事件）、形态确定 |
| fire_hydrant | 检测器 ✓（两可，默认留） | 形态极清晰；状态监测仍由 road VLM 管线承担 |
| bus_station / phone_booth / barrier_fence / manhole_cover / road_mirror / mailbox / power_station | **两可 → 默认检测器**，批 2 按样本量与打框质量逐类复核 | 边界清晰检测器可行；业务状态由既有 road 管线/哨兵承担，检测器作定位补充 |

样本预期见盘点 §3.1 road 分布。状态：草案（road 打框批启用）。**v1 入训叶类数修正：33 → 31**（billboard 移出、sculpture 待定；manhole 井盖高频见 phenomenon 网格佐证）。

## phenomenon（区域类；phenomenon 网格为原图级聚类，按场景组织非目标组织——预期内）

| 叶类 | 粗类锚 | 图证 | 决定 |
|---|---|---|---|
| water_puddle | ET14 | ✅c04 停车场积水洼（16 簇仅此 1 簇显见） | 保留（区域框需 VLM/人工打，训练批二） |
| manhole_damage | ET17 | ✅井盖在 10+ 簇显见（615 样本主导） | 保留（叶类名 manhole_damage：只标破损/异常井盖，完好井盖不入——防负类洪泛，P3 细化） |
| oil_stain | ET13 | 未见（12 样本全库） | **观察类** |
| road_damage | ET18 | c02 疑似修补痕迹，弱 | 保留（样本 83，观察边缘） |
| scattered_garbage_zone | ET19（llm 锚"散落垃圾"） | 未见 | 保留（llm 补样 30，待 P3） |
| construction_zone | ET6 | c14 围挡远景 | 保留（与 construction_fence 关系：区域 vs 实体，规范 §4/实体双标） |

## state-carrier（观察类，n=2 形态确证）

| 叶类 | 锚 | 图证 | 决定 |
|---|---|---|---|
| charging_bike | llm"违规充电" | ✅c01 车棚飞线 + c02 插线板/充电器特写 | **观察类**（ET15 全库 1 事件；形态两型已记词典；样本到达前不入训练） |

## v1 叶类计数

**2026-09-25 归属总表合并后以附录 A 为准**（含 person 新增、facility 双重身份、区域现象建模待定）。原 v1 计数（33 叶类）已被附录 A 的执行体口径取代：D1 类目池 22 + D2 烟火 2 + D3 哨兵 11 + 证据链道具 8。

## 附录 A：检测器×类目归属总表（2026-09-25 自 jxl 提案合并，单一数据源迁入）

> 归属原则：大体量道具 → D1 全帧；细小道具 → D1 类目 + 区域放大模式（同一模型双尺度级联，不加模型数）；烟火 → D2；场景定性 → D4。来源：research/2026-09-25-检测器类目归属总表.md（s4 合并时修正计数并显式化 facility 双重身份）。

| 执行体 | 类目池 | 计数 | 实现状态 |
|---|---|---|---|
| **D1 主检测器**（统一动态目标 YOLO） | 已训：ebike/garbage_bag；批 2 直入：shared_bike/tricycle/car/carton/foam_box/garbage_pile/construction_fence/scaffolding/vendor_booth/stall_cart/piled_goods/**person(辅助证据类)**；挂验证 5：bicycle（混淆<15% 否则并 shared_bike）/plastic_loose/shop_display/tarp_covered/leaves_pile；观察候补 3：dead_animal/oil_stain/charging_bike | **22** | 已训 2（v0-mini）；批 2 等终态批准 |
| **D2 烟火检测器**（独立线+时序投票） | smoke/open_flame | 2 | 预研；公开模型起步可先行，兼野炊证据源；**补 ET10 突发情况能力空档** |
| **D3 哨兵机制**（GPS+状态对比） | billboard/sculpture（仅哨兵）+ trash_can/fire_hydrant/bus_station/phone_booth/barrier_fence/road_mirror/mailbox/power_station（**与 D1 双重身份**：批 2 后视 D1 定位类 + 哨兵状态监测）+ manhole_cover（两段式：D1/D3 检出 + D4 破损判定） | 11 | 预研；road 第二批启用 |
| **D4 VLM 服务**（非检测器） | 场景判定（钓鱼/野营/野炊成立性）/无边界目标描述/状态对比/仲裁 | — | 机制已验证（批 1 实用中） |
| **证据链道具**（预研，归属已定） | tent/parasol/hammock（D1 全帧）；fishing_rod/folding_chair/fish_basket（D1 区域放大模式）；grill（双模式）；smoke/open_flame（→D2） | 8 | 样本≈0，归属先定，样本到达即启 |
| **区域现象**（批 2 前定建模方式） | water_puddle（倾向 D4 VLM 后置）/manhole_damage（两段式）/road_damage/scattered_garbage_zone/construction_zone（待定） | 5 | 不入 D1 v1 范围 |

---
## 修订日志

- v1-草案（2026-09-25）：任务书 A 六页网格人审后修订——truck→car、motorbike→ebike 合并；tarp_covered/leaves/dead_animal/oil_stain/charging_bike 转观察类；manhole_damage 明确「只标异常井盖」；各叶类补图证簇 id。
- v0（2026-09-25）：桌面草案骨架。
