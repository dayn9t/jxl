"""目标类别 profile: 类别定义单一数据源, 驱动 det_mine/doubao_relabel 等 5 处映射."""

import tomllib
from pathlib import Path

from pydantic import BaseModel

# src/jxl/target.py → src/jxl → src → 项目根; 与 common.JXL_ASSERTS 同约定.
_TARGETS_DIR = Path(__file__).parent.parent.parent / "targets"


class TargetProfile(BaseModel):
    """类别 profile: 一处定义, 驱动 YOLOE/GDINO 文本/RF-DETR cls/VLM prompt/权重/输出 cls."""

    name: str
    yolo_text: str
    rfdetr_cls_id: int | None = None
    vlm_prompt: str
    weights: str | None = None
    """在役模型先验(可选): None=无 target 票(新类链), L1 共识框即标注."""
    target_cls: int | None = None
    """target 模型自身类表中目标类位次; 设定则过滤该类检出(抑制多类模型他类头串扰)."""
    output_cls_id: int = 0


def load_target(name: str, profile_path: Path | None = None) -> TargetProfile:
    """加载 profile: --target-profile 显式 > targets/<name>.toml 内置 > FileNotFoundError."""
    path = profile_path or _TARGETS_DIR / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"target profile 不存在: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    return TargetProfile(**data)
