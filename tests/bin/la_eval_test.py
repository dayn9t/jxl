"""la_eval 纯函数单测: 标注文本对比 + 聚合统计。"""
from __future__ import annotations

from jxl.bin.la_eval import ImageDiff, aggregate, compare_label_text


def test_compare_perfect() -> None:
    base = "0 0.5 0.5 0.2 0.2"
    r = compare_label_text(base, base, 0.5)
    assert r.n_base == 1 and r.n_la == 1 and r.matched == 1
    assert r.missed == [] and r.extra == []


def test_compare_missed_and_extra() -> None:
    # 基准 2 框, la 1 框匹配 + 1 框位置不同 → missed 1 + extra 1
    base = "0 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.1 0.1"
    la = "0 0.2 0.2 0.1 0.1\n0 0.5 0.5 0.1 0.1"
    r = compare_label_text(base, la, 0.5)
    assert r.n_base == 2 and r.n_la == 2
    assert r.matched == 1 and r.missed == [1] and r.extra == [1]


def test_compare_both_empty() -> None:
    # 双空(负样本一致)
    r = compare_label_text("", "", 0.5)
    assert r.n_base == 0 and r.n_la == 0 and r.matched == 0
    assert r.missed == [] and r.extra == []


def test_compare_la_all_extra() -> None:
    # 基准空(负样本) + la 有框 → la 全多检(假阳)
    r = compare_label_text("", "0 0.5 0.5 0.2 0.2", 0.5)
    assert r.n_base == 0 and r.n_la == 1
    assert r.matched == 0 and r.extra == [0]


def test_aggregate_metrics() -> None:
    diffs = [
        ImageDiff(stem="a", n_base=1, n_la=1, matched=1, missed=[], extra=[]),
        ImageDiff(stem="b", n_base=2, n_la=1, matched=1, missed=[1], extra=[]),
        ImageDiff(stem="c", n_base=0, n_la=1, matched=0, missed=[], extra=[0]),
    ]
    r = aggregate(diffs)
    assert r["images"] == 3
    assert r["perfect_images"] == 1
    assert r["base_boxes"] == 3
    assert r["la_boxes"] == 3
    assert r["matched"] == 2
    assert abs(r["recall"] - 2 / 3) < 1e-9
    assert abs(r["precision"] - 2 / 3) < 1e-9
    assert abs(r["f1"] - 2 / 3) < 1e-9


def test_aggregate_empty() -> None:
    r = aggregate([])
    assert r["images"] == 0
    assert r["precision"] == 0.0 and r["recall"] == 0.0
