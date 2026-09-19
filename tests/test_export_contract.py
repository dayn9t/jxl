"""export_yolo_with_contract 契约构造的纯函数测试（无需 YOLO/GPU）。

验证 CLI 产出的 DetectContract / ClassifyContract dict: schema 合法 + 经
embed/read round-trip。这是检测/属性分类线 e2e 的 Python 侧闭环证明
（真实模型导出 + ml-vision 推理见 ml-vision 的 #[ignore] 集成测试）。
"""

from pathlib import Path

import jsonschema
import onnx
import pytest
from onnx import helper

from jxl.bin.export_yolo_with_contract import build_detect_contract
from jxl.contract import embed_contract, load_schema, read_contract


def _make_empty_onnx(path: Path) -> None:
    graph = helper.make_graph([], "g", [], [])
    model = helper.make_model(graph, producer_name="test")
    onnx.save(model, str(path))


def test_build_detect_contract_schema_valid() -> None:
    names = {0: "person", 1: "car"}
    contract = build_detect_contract(names, imgsz=640, conf=0.25)
    # embed_contract 内部会用 schema 校验；这里直接校验断言合法
    jsonschema.validate(contract, load_schema())


def test_build_detect_contract_roundtrips(tmp_path: Path) -> None:
    onnx_path = tmp_path / "m.onnx"
    _make_empty_onnx(onnx_path)
    names = {0: "person", 1: "car"}
    contract = build_detect_contract(names, imgsz=640, conf=0.25)
    embed_contract(onnx_path, contract)
    assert read_contract(onnx_path) == contract
    # class_names 按 id 排序
    assert contract["class_names"] == ["person", "car"]
    assert contract["preprocess"]["common"]["input_size"] == {
        "height": 640,
        "width": 640,
    }
    assert contract["postprocess"]["output_format"] == "n_a_xyxy_confcls"


def test_build_detect_contract_orders_classes_by_id() -> None:
    # ultralytics names 的 key 顺序不保证；契约必须按 id 排序
    names = {2: "c", 0: "a", 1: "b"}
    contract = build_detect_contract(names, imgsz=320, conf=0.5)
    assert contract["class_names"] == ["a", "b", "c"]
    assert contract["preprocess"]["common"]["input_size"] == {
        "height": 320,
        "width": 320,
    }


def test_build_classify_contract_head_crop() -> None:
    from jxl.bin.export_yolo_with_contract import build_classify_contract

    contract = build_classify_contract(
        {0: "normal", 1: "covered"}, 224, {"head": {"ratio": 0.35}}
    )
    assert contract["task"] == "classify"
    assert contract["preprocess"]["common"]["input_size"] == {"height": 224, "width": 224}
    # ultralytics classify 官方预处理：shortest-edge resize + center crop
    assert contract["preprocess"]["common"]["resize"] == "center_crop"
    assert contract["preprocess"]["normalize"] is None
    assert contract["preprocess"]["crop"] == {"head": {"ratio": 0.35}}
    # 图内已 softmax —— 契约如实声明 probabilities
    assert contract["postprocess"]["output_format"] == "probabilities"
    assert contract["postprocess"]["num_classes"] == 2
    assert contract["class_names"] == ["normal", "covered"]
    # crop 契约（Task 1 的 schema 扩展）必须被 schema 接受
    jsonschema.validate(contract, load_schema())


def test_build_classify_contract_without_crop() -> None:
    from jxl.bin.export_yolo_with_contract import build_classify_contract

    contract = build_classify_contract({0: "a", 1: "b"}, 224, None)
    assert "crop" not in contract["preprocess"] or contract["preprocess"]["crop"] is None
    jsonschema.validate(contract, load_schema())


def test_parse_crop_spec_kinds() -> None:
    from jxl.bin.export_yolo_with_contract import parse_crop_spec

    assert parse_crop_spec("head:0.35") == {"head": {"ratio": 0.35}}
    assert parse_crop_spec("phone_context:0.5") == {"phone_context": {"expand": 0.5}}


def test_parse_crop_spec_rejects_bad_values() -> None:
    from jxl.bin.export_yolo_with_contract import parse_crop_spec

    with pytest.raises(ValueError, match="head ratio"):
        parse_crop_spec("head:0")
    with pytest.raises(ValueError, match="phone_context expand"):
        parse_crop_spec("phone_context:5")
    with pytest.raises(ValueError, match="unknown crop kind"):
        parse_crop_spec("torso:0.5")
