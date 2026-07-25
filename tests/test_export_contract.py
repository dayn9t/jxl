"""export_yolo_with_contract.build_detect_contract 的纯函数测试（无需 YOLO/GPU）。

验证 CLI 产出的 DetectContract dict: schema 合法 + 经 embed/read round-trip。
这是检测线 e2e 的 Python 侧闭环证明（真实模型导出 + ml-vision 推理见
ml-vision 的 #[ignore] 集成测试）。
"""

from pathlib import Path

import jsonschema
import onnx
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
