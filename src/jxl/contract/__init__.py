"""ModelContract 写入/读取——Python 侧零 typed 镜像，schema 全部来自 Rust 派生物。"""

from pathlib import Path
from typing import Any, cast

import jsonschema
import onnx
import orjson

_SCHEMA_PATH = Path(__file__).parent / "schema" / "model_contract.schema.json"
_METADATA_KEY = "ml.model_contract"


def load_schema() -> dict[str, Any]:
    """加载 commit 进 repo 的 ModelContract JSON Schema（Rust schemars 派生）。"""
    return cast(dict[str, Any], orjson.loads(_SCHEMA_PATH.read_bytes()))


def embed_contract(onnx_path: Path, contract: dict[str, Any]) -> None:
    """把契约写进 ONNX 的 metadata。schema 校验失败 → 抛错，不产出无契约 onnx。"""
    jsonschema.validate(contract, load_schema())
    model = onnx.load(str(onnx_path))
    # 移除同 key 旧值，避免重复（protobuf repeated field 不支持切片赋值，逐个 remove）
    for prop in list(model.metadata_props):
        if prop.key == _METADATA_KEY:
            model.metadata_props.remove(prop)
    entry = model.metadata_props.add()
    entry.key = _METADATA_KEY
    entry.value = orjson.dumps(contract).decode()
    onnx.save(model, str(onnx_path))


def read_contract(onnx_path: Path) -> dict[str, Any]:
    """从 ONNX 读回契约；缺 key → KeyError（fail-fast）。"""
    model = onnx.load(str(onnx_path))
    for p in model.metadata_props:
        if p.key == _METADATA_KEY:
            return cast(dict[str, Any], orjson.loads(p.value))
    raise KeyError(_METADATA_KEY)
