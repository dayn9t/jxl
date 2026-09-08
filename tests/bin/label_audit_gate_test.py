"""label_audit gate 子命令单测: 共识闸门判定逻辑."""

from jxl.bin.label_audit import STRONG_VALIDATORS, _supporters


def _v(x1: float, y1: float) -> list[float]:
    """构造归一化框 xyxy+conf(左上角给 (x1,y1), 尺寸 0.2x0.3)."""
    return [x1, y1, x1 + 0.2, y1 + 0.3, 0.5]


def test_supporters_counts_strong_and_weak() -> None:
    t = (0.4, 0.4, 0.6, 0.7, 1.0)
    validators = {
        "yoloe": [_v(0.4, 0.4)],
        "rfdetr": [],
        "gdino": [_v(0.41, 0.41)],
        "la": [],
    }
    supp = _supporters(t, validators, 0.5)
    assert supp == {"yoloe", "gdino"}
    assert len(supp & set(STRONG_VALIDATORS)) == 1


def test_weak_only_support_is_the_poison_signature() -> None:
    """毒框签名: 仅 gdino+la 支持, 零强验证器 → 被闸门挡下."""
    t = (0.9, 0.45, 0.98, 0.65, 1.0)
    validators = {
        "yoloe": [],
        "rfdetr": [],
        "gdino": [[0.9, 0.45, 0.98, 0.65, 0.34]],
        "la": [[0.9, 0.45, 0.98, 0.65, 1.0]],
    }
    supp = _supporters(t, validators, 0.5)
    assert supp == {"gdino", "la"}
    assert len(supp & set(STRONG_VALIDATORS)) == 0, "零强支持 = n001 毒框签名, 应进待审"


def test_no_iou_overlap_no_support() -> None:
    t = (0.0, 0.0, 0.2, 0.3, 1.0)
    validators = {"yoloe": [_v(0.8, 0.8)], "gdino": [_v(0.5, 0.5)]}
    assert _supporters(t, validators, 0.5) == set()
