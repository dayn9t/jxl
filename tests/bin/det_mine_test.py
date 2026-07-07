"""det_mine 工具函数单测（_parse_weights）。

backend（detect_gdino/detect_rfdetr）与 cascade 分流属 imperative shell，
依赖模型权重/GPU，靠手动集成验证（见 plan Task 8）。
"""
from __future__ import annotations

from jxl.bin.det_mine import _parse_weights


def test_parse_weights_basic() -> None:
    assert _parse_weights("rfdetr:0.4,gdino:0.35,yoloe:0.25") == {
        "rfdetr": 0.4,
        "gdino": 0.35,
        "yoloe": 0.25,
    }


def test_parse_weights_empty() -> None:
    assert _parse_weights("") == {}


def test_parse_weights_extra_commas() -> None:
    assert _parse_weights("rfdetr:0.4,,gdino:0.6,") == {"rfdetr": 0.4, "gdino": 0.6}


def test_parse_weights_negative() -> None:
    assert _parse_weights("yoloe:-0.1") == {"yoloe": -0.1}


def test_parse_weights_spaces() -> None:
    assert _parse_weights(" rfdetr : 0.4 , gdino : 0.6 ") == {"rfdetr": 0.4, "gdino": 0.6}
