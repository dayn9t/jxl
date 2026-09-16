"""doubao_arbitrate 单测: 仲裁判定纯函数(赞同/不赞同/豆包独有/双空) + pick 优先序 + CLI 接口.

不测真实豆包 API(无 key 环境): grounding 链路复用 doubao_relabel.ground_one 同款代码,
靠生产运行验证; 此处只锁仲裁判定语义(纯函数, Box 字面量构造, 零网络).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from jxl.bin.doubao_arbitrate import PICK_PRIORITY, app, arbitrate_image, pick_label_box
from jxl.det.hardmine import Box

_REP: Box = (0.1, 0.1, 0.5, 0.5, 0.9)
_RFDETR: Box = (0.12, 0.1, 0.52, 0.48, 0.8)
_YOLOE: Box = (0.1, 0.12, 0.48, 0.5, 0.7)
_AGREE: Box = (0.1, 0.1, 0.5, 0.5, 0.95)  # 与 pick 框(rfdetr) IoU≈0.86 ≥ 0.4
_FAR: Box = (0.6, 0.6, 0.9, 0.9, 0.9)  # 与任何位置零重叠
_EXTRA: Box = (0.7, 0.05, 0.95, 0.4, 0.8)  # 豆包独有位置(无共识位置支持)


def _pos() -> tuple[Box, dict[str, Box]]:
    """单共识位置: rfdetr+yoloe 两模型支持(≥2 满足多数语义)."""
    return _REP, {"rfdetr": _RFDETR, "yoloe": _YOLOE}


def test_arbitrate_confirmed() -> None:
    """赞同确认: 豆包框与位置 IoU≥0.4 → 通过, 标注框取优先序代表框."""
    arb = arbitrate_image([_pos()], [_AGREE], 0.4)
    assert arb.confirmed
    assert arb.label_boxes == [_RFDETR]  # pick 优先序 rfdetr 先于 yoloe
    assert arb.unapproved == []
    assert arb.doubao_only == []


def test_arbitrate_position_unapproved() -> None:
    """不赞同: 豆包框零重叠 → 无多数; 远处豆包框同时构成豆包独有位置."""
    arb = arbitrate_image([_pos()], [_FAR], 0.4)
    assert not arb.confirmed
    assert arb.label_boxes == []
    assert len(arb.unapproved) == 1
    v = arb.unapproved[0]
    assert v.box == _RFDETR
    assert v.supporters == ["rfdetr", "yoloe"]
    assert v.best_doubao_iou < 0.4
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


def test_cli_rerun_cleans_stale_confirmed_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重跑清陈旧: 上次确认标签与 _errors.jsonl 不残留(防旧确认流入下游合并)."""
    import orjson
    from PIL import Image

    from jxl.bin import doubao_arbitrate as da
    from jxl.bin.rmb_ground import Detection

    images = tmp_path / "imgs"
    images.mkdir()
    for stem in ("f1", "f2"):
        Image.new("RGB", (64, 64)).save(images / f"{stem}.jpg")
    consensus = tmp_path / "consensus"
    (consensus / "review").mkdir(parents=True)
    box = [0.1, 0.1, 0.5, 0.5, 1.0]
    rows = [
        {
            "image": f"{s}.jpg",
            "score": 0.5,
            "target_boxes": [],
            "validators": {"rfdetr": [box], "yoloe": [box]},
            "breakdown": {},
        }
        for s in ("f1", "f2")
    ]
    manifest = consensus / "review" / "manifest.jsonl"
    manifest.write_text(
        "\n".join(orjson.dumps(r).decode() for r in rows) + "\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    agree_det = [Detection(label="person", bbox=[0.1, 0.1, 0.5, 0.5], conf=1.0)]

    async def ok_ground(paths, base_url, api_key, model, prompt, concurrency):
        return [(p, agree_det, None) for p in paths]

    async def err_ground(paths, base_url, api_key, model, prompt, concurrency):
        return [(p, [], "ValueError: HTTP 500: boom") for p in paths]

    monkeypatch.setattr(da, "load_backend", lambda *a: ("http://gw.test/v1/", "k-t", "m-t"))
    args = [str(consensus), str(images), str(out), "--target", "person"]

    monkeypatch.setattr(da, "ground_all", ok_ground)
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert sorted(p.stem for p in (out / "confirmed" / "labels").glob("*.txt")) == ["f1", "f2"]
    assert not (out / "_errors.jsonl").exists()

    # 重跑: 清单缩为 f1 且 grounding 失败 → f1/f2 陈旧确认标签全清, _errors 重写
    manifest.write_text(orjson.dumps(rows[0]).decode() + "\n", encoding="utf-8")
    monkeypatch.setattr(da, "ground_all", err_ground)
    r2 = CliRunner().invoke(app, args)
    assert r2.exit_code == 0, r2.output
    assert list((out / "confirmed" / "labels").iterdir()) == []
    errs = (out / "_errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(errs) == 1 and "HTTP 500" in errs[0]

    # 第三跑恢复无错 → _errors.jsonl 消失(错误清单不残留)
    monkeypatch.setattr(da, "ground_all", ok_ground)
    r3 = CliRunner().invoke(app, args)
    assert r3.exit_code == 0, r3.output
    assert [p.stem for p in (out / "confirmed" / "labels").glob("*.txt")] == ["f1"]
    assert not (out / "_errors.jsonl").exists()
