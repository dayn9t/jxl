"""标注项目元数据类型 —— rs/ulabel-core/src/label_meta.rs 的字段级镜像。

契约纪律见 ulabel docs/CONTRACT.md：真源在 Rust 侧，字段/形状变更须与
服务端 wire 契约同步修订。序列化约定：写侧 model_dump（CatProperty 经
serialize_by_alias 落盘 rs serde rename 的 "type" 键；zh/border_extend 等
None 字段用 exclude_none 省略，对应 skip_serializing_if）；读侧带
serde(default) 的字段给 Python 默认值（宽容旧文件）。

类型域声明：本模块全部为【标注域】类型（类别/属性 schema，服务多边形标注）。
检测域类型（rect 矩形、无属性表，如 jxl.det.d2d.D2dObject）禁止进入本包。

辨析：jxl 现役 meta（另一套 LabelMeta 形状）在 ``jxl.label.meta_jxl``，
两者并存不同模块，import 时显式区分（见 meta_jxl 模块 docstring）。
"""

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ShapeConfig(BaseModel):
    """形状显示配置。"""

    title_style: int
    """标题样式。"""
    thickness: int
    """线宽。"""
    auto_save: bool = True
    """是否自动保存（rs serde(default = "default_true")）。"""
    vertex_radius: float = 10.0
    """顶点半径（rs serde(default = "default_vertex_radius")）。"""


class RoiConfig(BaseModel):
    """ROI 配置。"""

    color: str
    """显示颜色。"""


class LocalizedNames(BaseModel):
    """本地化名称：`en` 与 `zh` 平级，无主次。

    `en` 兼任导出/互操作标识（必须存在）；`zh` 为中文显示名（缺失时回退 `en`）。
    """

    en: str
    """英文规范名（导出名）。"""
    zh: str | None = None
    """中文显示名（可选；rs serde(default) + skip_serializing_if）。"""


class ValueDef(BaseModel):
    """属性值定义。"""

    id: int
    """值 ID。"""
    names: LocalizedNames
    """本地化名称。"""
    description: str
    """描述。"""
    hotkey: str
    """快捷键。"""
    color: str
    """显示颜色。"""
    sign: str
    """标记符号。"""


class BorderExtend(BaseModel):
    """分类器裁剪边界扩展比例。

    从检测框裁取属性分类器输入时的四边外扩比例：`left`/`right` 相对检测框宽度，
    `top`/`bottom` 相对高度（bbox-crop-expansion 规范，常用 0.1 = 各边外扩 10%）。
    负值 = 内缩。逐边独立配置。与 jxl `PropMeta.border_extend` 同构。
    """

    left: FiniteFloat
    """向左扩展比例（相对宽度）；非有限值（nan/inf）构造即拒。"""
    right: FiniteFloat
    """向右扩展比例（相对宽度）；非有限值（nan/inf）构造即拒。"""
    top: FiniteFloat
    """向上扩展比例（相对高度）；非有限值（nan/inf）构造即拒。"""
    bottom: FiniteFloat
    """向下扩展比例（相对高度）；非有限值（nan/inf）构造即拒。"""


class PropDef(BaseModel):
    """属性类型定义。"""

    id: int
    """属性 ID。"""
    names: LocalizedNames
    """本地化名称。"""
    description: str
    """描述。"""
    values: list[ValueDef]
    """可选值列表。"""
    border_extend: BorderExtend | None = None
    """分类器裁剪边界扩展（可选；None/省略 = 不扩展，直接贴检测框）。"""


class CatProperty(BaseModel):
    """类别属性引用。"""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    """落盘键为 rs serde rename 的 "type"（serialize_by_alias）；构造时两键皆可。"""

    id: int
    """属性 ID。"""
    names: LocalizedNames
    """本地化名称。"""
    property_type: str = Field(alias="type")
    """属性类型。"""


class CatDef(BaseModel):
    """类别定义。"""

    id: int
    """类别 ID。"""
    names: LocalizedNames
    """本地化名称。"""
    description: str
    """描述。"""
    hotkey: str
    """快捷键。"""
    color: str
    """显示颜色。"""
    properties: list[CatProperty] = Field(default_factory=list)
    """关联的属性列表（rs serde(default)）。"""


type SpecialValue = ValueDef
"""特殊值定义（标注协议历史名称）。

rs 侧为 `pub type SpecialValue = ValueDef` 的类型别名（single-source-of-truth
合并，j-design-principles #8）；消费侧同样只做别名，保留
`LabelMeta.property_special_values` 字段的语义标签。
"""


class LabelMeta(BaseModel):
    """标注项目元数据。"""

    id: int
    """项目 ID。"""
    name: str
    """项目名称。"""
    description: str
    """描述。"""
    shape: ShapeConfig
    """形状配置。"""
    roi: RoiConfig
    """ROI 配置。"""
    categories: list[CatDef]
    """类别定义列表。"""
    property_types: list[PropDef]
    """属性类型列表。"""
    property_special_values: list[SpecialValue] = Field(default_factory=list)
    """特殊值列表（rs serde(default)，可省略）。"""
