# s4det 形态词典 v0（草案，2026-09-25）

> 文本源推导版（misc.md/road prompt/scene 203 词频/平台 TEXT 描述），**未经图证**——任务书 A 簇代表网格回来后逐族充实（⬜ 标记处），核实一条转正一条（skill j-consensus-labeling ⑪：词典与提示词共同演化）。
> 本文件管「每叶类长什么样、怎么辨」；类目归属与映射归 taxonomy.md。

## 1. vehicle 族

| 叶类草案 | 形态要点 | 图证 |
|---|---|---|
| ebike | 两轮+踏板位；外卖型带尾箱/餐箱（平台码 11 目击主形态）；夜间车尾灯/车牌可见 | ⬜ |
| bicycle | 细轮、车筐常见；与 shared_bike 区分=非统一涂装 | ⬜ |
| shared_bike | 统一涂装（蓝/黄/青桔品牌色）+ 车筐 + 扫码锁块；成排停放高发 | ⬜ |
| tricycle | 三轮；载货斗/封闭棚两形态；misc.md 排除项注意：破损锈蚀严重时视作垃圾 | ⬜ |
| car / truck / bus | 机动车；parking prompt 分汽车/卡车/公交车 | ⬜ |
| motorbike | 摩托车；与 ebike 边界=无踏板（拿不准入 hardcase） | ⬜ |

## 2. garbage 族（203 词频直证形态优先级）

| 叶类草案 | 形态要点 | 图证 |
|---|---|---|
| garbage_bag | 袋装：黑色/白色/橙色垃圾袋 + **编织袋**（203 词频 3106，最高单形态）+ 吨袋；TEXT 直证：「黑色垃圾袋」「橙色大垃圾袋」「灰白色大编织袋」「白色大型吨袋装垃圾」 | ⬜ |
| foam_box | 泡沫箱：TEXT 高频形态（「白色泡沫箱」×6+）；常与袋组合出现 | ⬜ |
| carton | 纸箱/纸板（词频 纸箱 1573/纸板 175）：「废弃纸箱」「纸箱和塑料筐」 | ⬜ |
| furniture_fallback | 废弃家具/床垫/木材/木板（词频 木 805/板 775）：「褐色木板」「黄色木箱」；misc.md：床垫、废旧家具、木材 | ⬜ |
| tarp_pile | **篷布覆盖堆**（词频 537；TEXT：「黑色篷布覆盖的堆放物」「粉色和黑色篷布」「蓝白条纹篷布」）——显著稳定形态，建议独立叶类；与 garbage_pile 关系=篷布覆盖时只见篷布，见不到内部个体 | ⬜ |
| plastic_loose | 筐/盆/桶散放（词频 桶 852/塑料筐 240/盆 104） | ⬜ |
| leaves_pile | 落叶/枯枝（词频 111/99） | ⬜ |
| dead_animal | 动物尸体（misc.md 明文，样本预期极少） | ⬜ |
| garbage_pile | 密集堆体 aggregate（标注规范 §6）：个体叠压不可数时整体成框 | ⬜ |
| debris_misc | 杂物（词频 2501）兜底形态——聚类后若散则并回各叶类 | ⬜ |

## 3. vendor/occupy 族

| 叶类草案 | 形态要点 | 图证 |
|---|---|---|
| stall_cart | 流动摊车/售卖车（misc.md：摆地摊或流动售卖车摊位） | ⬜ |
| vendor_booth | 地摊阵/摊位（含铺布陈列） | ⬜ |
| shop_display | 店外经营载体：店门口货架/商品陈列/外摆桌椅 | ⬜ |
| construction_fence | 施工围挡（词频 385；TEXT：「黄色围挡及燃气施工提示牌」「木质围挡」）；高度/颜色多样，蓝白铁皮常见 | ⬜ |
| scaffolding | 脚手架（TEXT：「外墙搭建黄色脚手架」） | ⬜ |
| piled_goods | 可辨堆物个体（整齐堆放，区分于垃圾：misc.md「整齐=堆物、凌乱=垃圾」） | ⬜ |

## 4. facility 族（road 部件 + scene 202 目标）

trash_can（分类垃圾桶，金属方箱四色+标识；**scene 202 数据原文「垃圾统」=平台侧笔误，同此物**）/ fire_hydrant（红色铸铁立柱+法兰口）/ bus_station（站牌+候车棚两组件）/ phone_booth（封闭/开放两形态）/ barrier_fence（机非隔离栏，立柱+横杆，绿白灰黄）/ manhole_cover（井盖：圆形铸铁纹路）/ road_mirror（ convex 反光镜）/ mailbox（邮筒）/ power_station（配电站箱体）/ sculpture / billboard（宣传画，墙面附着）。各自排除项详见 deploy/event/template/road/*.md（单一数据源，此处不复制）。

## 5. phenomenon 族（区域画法，规范 §4）

water_puddle / oil_stain / road_damage（坑洼裂缝）/ scattered_garbage_zone（≥1㎡ 散布区）/ construction_zone（围挡围出区域）。判据要点：框现象边界非实体轮廓；现象内实体另标。

## 6. state-carrier 族

charging_bike（ebike+飞线/充电器；聚类验证是否成立）。

---
## 修订日志

- v0（2026-09-25）：文本源草案，全部 ⬜ 待图证。
