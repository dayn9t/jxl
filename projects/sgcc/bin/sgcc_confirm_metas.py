"""SGCC 确认集 meta 单一数据源——每种确认标注类型的 LabelMeta 构造器.

每个确认集自包含 meta.json5（服务/前端/UI 全部由它驱动），本模块是这些 meta
的唯一定义处：建集脚本（build_role_confirm_set.py）与项目初始化
（init_confirm_project.py）都从这里取，禁止各自手搓。

值域统一遵循台账 #17 通用原则：标准选项 + 排除类（undecidable 不可判定）
+ 出错类（detector_error 上级检测器错误通道——仅在被审属性与框/类别正交时设，
框质量本身就是审查对象的类型（bad_fit）不重复设）。
"""

from vlabel.meta import (
    CatDef,
    CatProperty,
    LabelMeta,
    LocalizedNames,
    PropDef,
    RoiConfig,
    ShapeConfig,
    ValueDef,
)

# ---- 共用构造件 -------------------------------------------------------------

PERSON_CAT_ID = 0


def _value(value_id: int, en: str, zh: str, hotkey: str, color: str, sign: str) -> ValueDef:
    return ValueDef(
        id=value_id,
        names=LocalizedNames(en=en, zh=zh),
        description=zh,
        hotkey=hotkey,
        color=color,
        sign=sign,
    )


def _meta(meta_id: int, name: str, description: str, prop_type: str, prop: PropDef) -> LabelMeta:
    """单属性确认集的统一外形：person 类别 + 一个 PropDef."""
    return LabelMeta(
        id=meta_id,
        name=name,
        description=description,
        shape=ShapeConfig(title_style=1, thickness=2),
        roi=RoiConfig(color="#800080"),
        categories=[
            CatDef(
                id=PERSON_CAT_ID,
                names=LocalizedNames(en="person", zh="人员"),
                description="人员",
                hotkey="1",
                color="#FF0000",
                properties=[
                    CatProperty(
                        id=prop.id,
                        names=prop.names,
                        property_type=prop_type,
                    )
                ],
            )
        ],
        property_types=[prop],
    )


def _prop(prop_id: int, en: str, zh: str, description: str, values: list[ValueDef]) -> PropDef:
    return PropDef(
        id=prop_id,
        names=LocalizedNames(en=en, zh=zh),
        description=description,
        values=values,
    )


# ---- role 属性确认（首航，id 1006） ------------------------------------------

# role_n v3 词典（字母序权威，见 jxl research 2026-09-14-role-v3-对接预演）
# + #17 原则两值（undecidable / detector_error）
ROLE_VALUES: list[tuple[int, str, str, str, str, str]] = [
    # id, en, zh, hotkey, color, sign — staff 系橙黄可区分变体，客户绿，
    # not_person 灰，undecidable 紫，detector_error 红
    (0, "cleaner", "保洁", "1", "#F5C542", "CL"),
    (1, "customer", "客户", "2", "#2CA02C", "CU"),
    (2, "leader", "负责人", "3", "#F0A202", "LD"),
    (3, "manager", "经理", "4", "#E07B00", "MG"),
    (4, "not_person", "非人员", "5", "#808080", "NP"),
    (5, "security", "安保", "6", "#C46A10", "SC"),
    (6, "teller", "柜员", "7", "#FFDA7A", "TL"),
    (7, "undecidable", "不可判定", "8", "#9467BD", "UD"),
    (8, "detector_error", "检测出错", "9", "#D62728", "DE"),
]
ROLE_IDS: dict[str, int] = {en: value_id for value_id, en, *_ in ROLE_VALUES}
CUSTOMER_ID = ROLE_IDS["customer"]


def role_meta() -> LabelMeta:
    return _meta(
        1006,
        "角色确认",
        "sgcc-n001 人员 role 属性复审确认集（field trial）",
        "role",
        _prop(
            0,
            "role",
            "人员角色",
            "人员角色（v3 词典 7 值 + 不可判定 + 检测出错）",
            [_value(*v) for v in ROLE_VALUES],
        ),
    )


# ---- 框质量确认（bad_fit 复审，id 1007） --------------------------------------
# 框质量本身是审查对象——「框错」由标准值 box_broken 表达，不再设 detector_error。

def bad_fit_meta() -> LabelMeta:
    return _meta(
        1007,
        "框质量复核",
        "sgcc-n001 person bad_fit 弃帧复审：分离可回收边缘截断与真不可信框",
        "box_quality",
        _prop(
            0,
            "box_quality",
            "框质量",
            "检测框质量：框可信 / 边缘截断（可回收）/ 不可信框 / 不可判定",
            [
                _value(0, "box_ok", "框可信", "1", "#2CA02C", "✓"),
                _value(1, "edge_truncated", "边缘截断可回收", "2", "#FF7F0E", "◧"),
                _value(2, "box_broken", "框不可信", "3", "#D62728", "✗"),
                _value(3, "undecidable", "不可判定", "4", "#800080", "?"),
            ],
        ),
    )


# ---- 是否真人（not_person 检出复审，id 1008） ----------------------------------

def person_verify_meta() -> LabelMeta:
    return _meta(
        1008,
        "人物核验",
        "sgcc-n001 not_person 检出复审：框内是否真人",
        "is_person",
        _prop(
            0,
            "is_person",
            "是否真人",
            "框内目标是否真人；detector_error = 框/类别本身出错（反馈上级）",
            [
                _value(0, "person", "真人", "1", "#2CA02C", "人"),
                _value(1, "not_person", "非人", "2", "#808080", "非"),
                _value(2, "undecidable", "不可判定", "3", "#800080", "?"),
                _value(3, "detector_error", "检测出错", "4", "#8B0000", "!"),
            ],
        ),
    )


# ---- 上半身可见性（upper 双门不一致复审，id 1009） ------------------------------

def upper_visibility_meta() -> LabelMeta:
    return _meta(
        1009,
        "上半身可见性",
        "sgcc-n001 upper_body_visible 双门不一致复审",
        "upper_body_visible",
        _prop(
            0,
            "upper_body_visible",
            "上半身可见",
            "目标上半身是否可见；detector_error = 框/类别本身出错（反馈上级）",
            [
                _value(0, "visible", "可见", "1", "#2CA02C", "见"),
                _value(1, "not_visible", "不可见", "2", "#808080", "遮"),
                _value(2, "undecidable", "不可判定", "3", "#800080", "?"),
                _value(3, "detector_error", "检测出错", "4", "#8B0000", "!"),
            ],
        ),
    )


# ---- reid 对确认（osnet rejected 对复审，id 1010） ------------------------------
# 建集时把一对裁片合成单图（并排），审查「是否同一人」。

def reid_pair_meta() -> LabelMeta:
    return _meta(
        1010,
        "再识别配对",
        "sgcc-n001 osnet 重连分歧对复审：两裁片是否同一人",
        "pair_same",
        _prop(
            0,
            "pair_same",
            "是否同人",
            "并排两裁片是否同一人；detector_error = 裁片跑偏/框错（反馈上级）",
            [
                _value(0, "same_person", "同人", "1", "#2CA02C", "同"),
                _value(1, "different_person", "异人", "2", "#D62728", "异"),
                _value(2, "undecidable", "不可判定", "3", "#800080", "?"),
                _value(3, "detector_error", "检测出错", "4", "#8B0000", "!"),
            ],
        ),
    )
