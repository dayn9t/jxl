<!-- TODO(relocate): README 待重写反映新现状（训练/导出/标注/ModelContract），见 SPEC §5.5 -->
# jxl — Python ML 库

dayn9t 推理层的 Python 侧：模型训练 / 导出 / 标注工具 + `ModelContract` 写入端（把训练侧元数据 embed 进 ONNX，驱动 Rust 侧 `ml-vision` 推理契约）。

> 旧版 README（2022 shtm/trash 操作指南、ias 脚本流程）已废弃——本文件只描述 jxl **库**本身。

## 结构（`src/jxl/`）

| 模块 | 职责 |
|------|------|
| `det/` | 检测（YOLO 集成、`Detector` trait、mm） |
| `seg/` | 分割（SAM、mask） |
| `track/` | 跟踪（`IouTracker`） |
| `cls/` | 分类（`IClassifier[T]`、`ClassifierRes` Protocol） |
| `label/` | 标注工具与格式转换（COCO / Darknet / KITTI / labelme，含 `DataCoco`、tile、blend） |
| `contract/` | **ModelContract 写入端**——`embed_contract` 把 schema 写入 ONNX metadata key `ml.model_contract`（schema 权威在 `ml-types::ModelContract` + schemars，见跨 repo spec） |
| `model/` | 模型类型（`ModelInfo[OptT]` 泛型） |
| `bin/` | CLI 入口（`jml_label` / `jml_prop` / `jml_sample` / 标注审核等，经 `uv run` 或 entry point 调用） |
| `io/` `iqa/` `od/` `sam/` `util/` `yolo/` | I/O、图像质量、目标检测、SAM、工具、YOLO |

`examples/oai/` 下为 LLM provider demo 脚本（不在可 import 包命名空间）。

## ModelContract 契约

jxl 是训练↔推理 artifact 契约的**写入侧**：

```
jxl (embed_contract)  →  ONNX metadata[ml.model_contract]  →  ml-vision (parse_contract, fail-fast)
                              ↑ schema 权威
                      ml-types::ModelContract (schemars 派生)
```

`ml-types/tests/schema_sync.rs` 守护 jxl 的 schema 拷贝与 ml-types 派生 schema 一致（drift 即测试失败）。

## 开发

```bash
uv sync                 # 安装
uv run ruff check .     # lint（gate：精选规则，见 ruff.toml）
uv run mypy .           # 类型检查（strict=false 渐进基线 + 高价值告警）
uv run pytest           # 测试
```

## 关键约束

- **类型严格**：全量注解、禁 `Any`/`hasattr`/裸 `except`（j-python-strict）；`BaseModel>dict`、`@dataclass(frozen=True, slots=True)`、`StrEnum`。
- **契约一致**：改 `ml-types::ModelContract` schema 后须重新生成 schema（`ml-types` 的 `gen_schema`）并同步本库 `contract/schema/` 拷贝，否则 `schema_sync` 失败。
