"""s4det 任务书 A 共享件：数据模型 / 五粗类抽样规格 / 隔离台账助手.

单一数据源:
- 族抽样规格 = 本文件 FAMILY_SPECS(任务书 A 抽样表的代码化)
- EventType 中文名 = s4 仓 rs/s4-t/src/mobile_event.rs 枚举的镜像(注释指向真源)
- crop 外延规范 = skill bbox-crop-expansion(执行在 a_crop.py, 经 vlabel.crop 标准函数)
"""

from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BATCH_ID = "2026-09-25-p0-cluster"
"""本次跑批批次号(隔离台账 batch 字段与产物目录对齐)."""

ET_NAMES: dict[int, str] = {
    1: "机动车乱停",
    2: "非机动车乱停",
    3: "丢弃垃圾",
    4: "道路设施损坏",
    5: "店招损坏",
    6: "占道施工",
    7: "店铺装修",
    8: "店招消失",
    9: "店招新增",
    10: "突发情况",
    11: "占道堆物",
    12: "占道设摊",
    13: "地面油污",
    14: "地面积水",
    15: "违规充电",
    16: "占道经营",
    17: "井盖破损",
    18: "道路破损",
    19: "散落垃圾",
}
"""镜像自 s4 仓 EventType 枚举(research §2.6); 真源 = rs/s4-t/src/mobile_event.rs."""

CONF_BANDS: tuple[str, ...] = ("<50", "50-89", ">=90")
"""平台框置信度分档(任务书 A: Conf<50 保留入聚类但记录分档统计)."""

GRID_CODE12_BAND = "12"
"""网格中单独成列的平台码(CODEBOOK 待定码 12, 人审一锤定音)."""


def conf_band(conf: float) -> str:
    """置信度 → 分档标签."""
    if conf < 50:
        return "<50"
    return "50-89" if conf < 90 else ">=90"


def code_band(desc: str | None) -> str:
    """平台 Description → 码分档: 纯数字码 / "11,[r,c].." 取逗号前码 / 自由文本 text / 空 none."""
    if desc is None or not desc.strip():
        return "none"
    head = desc.split(",", 1)[0].strip()
    return head if head.isdigit() else "text"


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class LlmPair(BaseModel):
    """llm_entries 单 object 双模型分类(语义标签候选)."""

    doubao: str | None = None
    qwen: str | None = None
    agree: bool | None = None


class ObjectRec(BaseModel):
    """平台框(归一化 4 点外接矩形)."""

    object_id: str
    code: str
    band: str
    conf: float
    conf_band: str
    bbox: list[float]  # x1, y1, x2, y2 归一化


class EventRec(BaseModel):
    """事件索引行(event_index.jsonl)."""

    dir: str
    date: str
    event_id: str
    event_type: int
    et_name: str
    device_id: str
    hour: int
    image_path: str | None
    objects: list[ObjectRec]
    llm: dict[str, LlmPair]

    @property
    def month(self) -> str:
        return self.date[:7]


class ManifestRow(BaseModel):
    """抽样 manifest 行(每行 = 一张 crop 或一张原图样本)."""

    family: str
    mode: Literal["crop", "original"]
    event_dir: str
    date: str
    event_id: str
    rid: str
    event_type: int
    et_name: str
    device_id: str
    hour: int
    object_id: str | None = None
    code: str | None = None
    code_band: str = "none"
    conf: float | None = None
    conf_band: str | None = None
    bbox: list[float] | None = None
    llm_doubao: str | None = None
    llm_qwen: str | None = None
    llm_agree: bool | None = None
    source: str = "event_type"
    # 行来源: event_type=按事件类型抽样 | llm_class:<类别>=llm 一致类别补样 | no_object=无框原图
    image_path: str
    crop_path: str


class QuarantineEntry(BaseModel):
    """隔离台账行(j-quarantine-ledger 最小字段集)."""

    id: str
    batch: str
    gate: str
    verdict: str
    model: str
    priority: Literal["high", "low"] = "low"
    recoverable: bool = True


class FamilySpec(BaseModel):
    """粗类抽样规格(任务书 A 抽样表)."""

    name: str
    event_types: list[int]
    mode: Literal["crop", "original"]
    target: int
    full_types: list[int] = []
    # 小类全量(免摄像头去过密); 其余 rare 类型超 rare_cap 时按月均匀降采样
    rare_cap: int | None = None
    max_objs: int = 3
    fallback_original: bool = False
    # crop 族: 无框事件是否用原图顶(sample 上限 = target 的 10%)
    # original 族: 是否为有框事件追加 object crop 行(state_carrier)
    with_object_crops: bool = False
    llm_class_boost: dict[str, int] = {}
    # ET15/16/19 在 event.json 无事件(0 条), 仅以 llm_entries 双模型一致类别存在于
    # 其他事件的对象上(2026-09-25 全量扫描实证)——按类别从全库补样; 值 = 补样上限


FAMILY_SPECS: dict[str, FamilySpec] = {
    spec.name: spec
    for spec in [
        FamilySpec(
            name="vehicle", event_types=[2, 1], mode="crop", target=1500, rare_cap=400
        ),
        FamilySpec(
            name="garbage",
            event_types=[3, 11, 19],
            mode="crop",
            target=1500,
            rare_cap=400,
            llm_class_boost={
                "散落垃圾": 100
            },  # ET19 散落垃圾: 事件 0 条, llm 一致对象 31 个
        ),
        FamilySpec(
            name="vendor_occupy",
            event_types=[12, 16, 6, 7],
            mode="crop",
            target=1000,
            rare_cap=350,
            fallback_original=True,
            llm_class_boost={
                "占道经营": 150
            },  # ET16 占道经营: 事件 0 条, llm 一致对象 122 个
        ),
        FamilySpec(
            name="phenomenon",
            event_types=[13, 14, 17, 18],
            mode="original",
            target=1000,
            full_types=[13, 18],
        ),
        FamilySpec(
            name="state_carrier",
            event_types=[15],
            mode="original",
            target=10**9,
            with_object_crops=True,
            llm_class_boost={
                "违规充电": 10**9
            },  # ET15 违规充电: 事件 0 条, llm 一致对象仅 1 个(全量)
        ),
    ]
}

# ---------------------------------------------------------------------------
# io 助手
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """读 jsonl(空文件返回空表)."""
    if not path.is_file():
        return []
    return [
        orjson.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    """写 jsonl(建父目录)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\n".join(orjson.dumps(r.model_dump()) for r in rows)
    path.write_bytes(payload + b"\n" if payload else b"")


def append_quarantine(ledger: Path, entries: list[QuarantineEntry]) -> int:
    """追加隔离台账(建父目录); 追加语义不清理历史, 按 id 去重防重跑重复落账."""
    if not entries:
        return 0
    ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = {str(r.get("id")) for r in read_jsonl(ledger)}
    payload = b"".join(
        orjson.dumps(e.model_dump()) + b"\n" for e in entries if e.id not in seen
    )
    if not payload:
        return 0
    with ledger.open("ab") as f:
        f.write(payload)
    return payload.count(b"\n")
