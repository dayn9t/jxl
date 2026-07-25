from pathlib import Path

import onnx
import pytest
from onnx import helper

from jxl.contract import embed_contract, load_schema, read_contract


def _make_empty_onnx(path: Path) -> None:
    # 最小合法 ONNX（一个空图），用于测试 metadata 读写
    graph = helper.make_graph([], "g", [], [])
    model = helper.make_model(graph, producer_name="test")
    onnx.save(model, str(path))


def test_roundtrip(tmp_path: Path) -> None:
    onnx_path = tmp_path / "m.onnx"
    _make_empty_onnx(onnx_path)
    contract = {
        "task": "detect",
        "meta": {"schema_version": 1, "exported_by": "ultralytics"},
        "preprocess": {
            "common": {
                "input_size": {"height": 640, "width": 640},
                "resize": "letterbox",
                "color": "rgb",
                "dtype": "f32",
                "layout": "nchw",
            },
            "scale": 0.003921569,
        },
        "postprocess": {"conf_threshold": 0.25, "output_format": "n_a_xyxy_confcls"},
        "class_names": ["person", "car"],
    }
    embed_contract(onnx_path, contract)
    assert read_contract(onnx_path) == contract


def test_reject_bad_contract(tmp_path: Path) -> None:
    onnx_path = tmp_path / "m.onnx"
    _make_empty_onnx(onnx_path)
    # 缺 postprocess → schema 校验失败
    bad = {
        "task": "detect",
        "meta": {"schema_version": 1, "exported_by": "ultralytics"},
    }
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        embed_contract(onnx_path, bad)


def test_missing_key_raises(tmp_path: Path) -> None:
    onnx_path = tmp_path / "m.onnx"
    _make_empty_onnx(onnx_path)
    with pytest.raises(KeyError):
        read_contract(onnx_path)


def test_schema_loads() -> None:
    s = load_schema()
    assert isinstance(s, dict)


def _detect_contract(conf: float) -> dict:
    return {
        "task": "detect",
        "meta": {"schema_version": 1, "exported_by": "ultralytics"},
        "preprocess": {
            "common": {
                "input_size": {"height": 640, "width": 640},
                "resize": "letterbox",
                "color": "rgb",
                "dtype": "f32",
                "layout": "nchw",
            },
            "scale": 0.003921569,
        },
        "postprocess": {"conf_threshold": conf, "output_format": "n_a_xyxy_confcls"},
        "class_names": ["person", "car"],
    }


def test_embed_is_idempotent(tmp_path: Path) -> None:
    """重复 embed 同一 onnx：read 返回最新值，metadata 不残留旧 key（幂等）。"""
    onnx_path = tmp_path / "m.onnx"
    _make_empty_onnx(onnx_path)
    embed_contract(onnx_path, _detect_contract(0.25))
    embed_contract(onnx_path, _detect_contract(0.5))  # 第二次 embed（不同 conf）

    assert read_contract(onnx_path) == _detect_contract(0.5)  # 最新值生效
    # 幂等：只有一个 ml.model_contract 条目（去重逻辑移除旧值）
    m = onnx.load(str(onnx_path))
    keys = [p.key for p in m.metadata_props if p.key == "ml.model_contract"]
    assert len(keys) == 1
