"""doubao_arbitrate 单测: 仲裁判定纯函数(赞同/不赞同/豆包独有/双空) + pick 优先序 + CLI 接口.

不测真实豆包 API(无 key 环境): grounding 链路复用 doubao_relabel.ground_one 同款代码,
靠生产运行验证; 此处只锁仲裁判定语义(纯函数, Box 字面量构造, 零网络).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jxl.bin.doubao_arbitrate import PICK_PRIORITY, app, arbitrate_image, pick_label_box
from jxl.det.hardmine import Box

_REP: Box = (0.1, 0.1, 0.5, 0.5, 0.9)
_RFDETR: Box = (0.12, 0.1, 0.52, 0.48, 0.8)
_YOLOE: Box = (0.1, 0.12, 0.48, 0.5, 0.7)
_AGREE: Box = (0.1, 0.1, 0.5, 0.5, 0.95)  # 与 pick 框(rfdetr) IoU≈0.86 ≥ 0.5
_FAR: Box = (0.6, 0.6, 0.9, 0.9, 0.9)  # 与任何位置零重叠
_EXTRA: Box = (0.7, 0.05, 0.95, 0.4, 0.8)  # 豆包独有位置(无共识位置支持)


def _pos() -> tuple[Box, dict[str, Box]]:
    """单共识位置: rfdetr+yoloe 两模型支持(≥2 满足多数语义)."""
    return _REP, {"rfdetr": _RFDETR, "yoloe": _YOLOE}


def test_arbitrate_confirmed() -> None:
    """赞同确认: 豆包框与位置 IoU≥0.5 → 通过, 标注框取优先序代表框."""
    arb = arbitrate_image([_pos()], [_AGREE], 0.5)
    assert arb.confirmed
    assert arb.label_boxes == [_RFDETR]  # pick 优先序 rfdetr 先于 yoloe
    assert arb.unapproved == []
    assert arb.doubao_only == []


def test_arbitrate_position_unapproved() -> None:
    """不赞同: 豆包框零重叠 → 无多数; 远处豆包框同时构成豆包独有位置."""
    arb = arbitrate_image([_pos()], [_FAR], 0.5)
    assert not arb.confirmed
    assert arb.label_boxes == []
    assert len(arb.unapproved) == 1
    v = arb.unapproved[0]
    assert v.box == _RFDETR
    assert v.supporters == ["rfdetr", "yoloe"]
    assert v.best_doubao_iou < 0.5
    assert arb.doubao_only == [_FAR]


def test_arbitrate_doubao_only_position() -> None:
    """豆包独有位置: 共识位置获赞同但豆包另有无人支持的框 → 仍无多数."""
    arb = arbitrate_image([_pos()], [_AGREE, _EXTRA], 0.5)
    assert not arb.confirmed
    assert arb.unapproved == []  # 位置本身获赞同
    assert arb.doubao_only == [_EXTRA]


def test_arbitrate_both_empty_confirmed() -> None:
    """双空确认: 无共识位置且豆包无框 → 空标确认(label_boxes=[])."""
    arb = arbitrate_image([], [], 0.5)
    assert arb.confirmed
    assert arb.label_boxes == []
    assert arb.unapproved == []
    assert arb.doubao_only == []


def test_arbitrate_one_doubao_box_cannot_support_two_positions() -> None:
    """一对一贪心: 单豆包框与两位置均重叠也只算赞同一个, 另一位置未获赞同."""
    p1 = ((0.1, 0.1, 0.5, 0.5, 0.9), {"rfdetr": (0.1, 0.1, 0.5, 0.5, 0.8)})
    p2 = ((0.12, 0.08, 0.52, 0.48, 0.85), {"gdino": (0.12, 0.08, 0.52, 0.48, 0.75)})
    arb = arbitrate_image([p1, p2], [(0.1, 0.1, 0.5, 0.5, 0.95)], 0.5)
    assert not arb.confirmed
    assert len(arb.unapproved) == 1
    assert arb.doubao_only == []


def test_pick_label_box_fixed_priority_over_conf() -> None:
    """固定优先序: la conf 恒 1.0 也不抢占 rfdetr(conf 0.3)."""
    la = (0.11, 0.11, 0.51, 0.51, 1.0)
    low = (0.12, 0.1, 0.52, 0.48, 0.3)
    assert pick_label_box(_REP, {"la": la, "rfdetr": low}) == low
    assert PICK_PRIORITY[:3] == ("rfdetr", "gdino", "yoloe")


def test_pick_label_box_unknown_validator_falls_back_to_representative() -> None:
    """支持者全为未知校验器名 → 回退代表框(不产生 None)."""
    unknown = (0.11, 0.11, 0.51, 0.51, 1.0)
    assert pick_label_box(_REP, {"foo": unknown}) == _REP


def test_cli_help_contains_iou_flags() -> None:
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "--iou-major" in r.output
    assert "--iou-consensus" in r.output


def test_cli_zero_review_rows_errors(tmp_path: Path) -> None:
    """零 review 行(豆包空输入) → 入口期报错退出, 不触 API."""
    consensus = tmp_path / "consensus"
    (consensus / "review").mkdir(parents=True)
    (consensus / "review" / "manifest.jsonl").write_text("\n", encoding="utf-8")
    r = CliRunner().invoke(
        app,
        [
            str(consensus),
            str(tmp_path / "imgs"),
            str(tmp_path / "out"),
            "--target",
            "person",
        ],
    )
    assert r.exit_code == 1
