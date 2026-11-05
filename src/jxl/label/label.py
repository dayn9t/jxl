"""标注文件类型 —— rs/ulabel-core/src/label.rs 的字段级镜像。

契约纪律见 ulabel docs/CONTRACT.md：真源在 Rust 侧，字段/形状变更须与服务端
wire 契约（OpenAPI 快照）同步修订。序列化约定：写侧用 model_dump(exclude_none=True)
（对应 rs skip_serializing_if）；读侧字段全默认宽容（对应 rs serde(default)，
前向兼容旧文件）。唯一例外：Label.created_at/last_modified 构造即拒 naive
（rs 真源 DateTime<Local> 只接受带 offset 的 RFC3339，接受域与 Rust 侧精确一致）。

类型域声明：本模块全部为【标注域】类型（多边形 + 属性表 + 备注）。
检测域类型（rect 矩形、无属性表，如 jxl.det.d2d.D2dObject）禁止进入本包。
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from jxl.label.geometry import Polygon


class PropertyEntry(BaseModel):
    """属性条目（单值）。"""

    id: int
    """属性 ID。"""
    value: int
    """属性值 ID。"""
    confidence: float
    """置信度 [0, 1]。"""
    note: str | None = None
    """备注（可选；None 序列化时省略）。"""


class Object(BaseModel):
    """标注目标。"""

    id: int
    """目标 ID。"""
    category: int
    """类别 ID。"""
    confidence: float
    """置信度 [0, 1]。"""
    polygon: Polygon
    """目标多边形。"""
    note: str | None = None
    """备注（可选；None 序列化时省略）。"""
    properties: list[PropertyEntry] = Field(default_factory=list)
    """附加属性列表。"""


class Label(BaseModel):
    """标注文件（单个样本）。"""

    version: str
    """协议版本（semver 字符串）。"""
    user_agent: str
    """创建工具标识。"""
    created_at: datetime
    """创建时间（必须 tz-aware；rs 真源为 chrono DateTime<Local>，pydantic 截断到微秒，见 ulabel docs/CONTRACT.md）。"""
    last_modified: datetime
    """最后修改时间（必须 tz-aware，同 created_at）。"""
    note: str | None = None
    """文件级备注（可选；None 序列化时省略）。"""
    rois: list[Polygon] = Field(default_factory=list)
    """ROI 多边形（可空）。"""
    objects: list[Object]
    """标注目标列表。"""

    @field_validator("created_at", "last_modified")
    @classmethod
    def _reject_naive_timestamps(cls, value: datetime) -> datetime:
        """构造即拒 naive——与 rs 真源接受域精确对齐。

        chrono `DateTime<Local>` 的 serde 只接受带 offset 的 RFC3339：naive
        值落盘后服务端（POST verdicts）与 GUI 均解析失败。非法状态在镜像
        侧即不可表示，而非等到 wire 边界 400。
        """
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("timestamp must be tz-aware (rs truth type is chrono DateTime<Local>; naive is rejected server-side)")
        return value
