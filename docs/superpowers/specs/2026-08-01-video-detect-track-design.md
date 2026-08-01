---
title: 视频检测与跟踪模块（vdt）设计 spec
project: py/jxl（Python 原型，本 spec 实现目标）+ next/ml（Rust 生产，移植暂缓）
date: 2026-08-01
status: design-approved, ready-for-impl
scope: detect → 双模 track（IoU/ReID）→ 条件性 pose，库核心 + CLI 批处理
audience: 在 py/jxl 新开会话、据此实现的工程师 / Claude
---

# 视频检测与跟踪模块（vdt）— 设计 spec

> **给新会话的读者**：本文自包含全部实现所需信息——模块目标、jxl 代码库现状与复用点、架构决策与理由、各组件协议/impl、ReID 与 pose 算法细节、模型栈与 ONNX 来源、错误处理、测试、配置/CLI、实现分期、Rust 移植映射。深度背景（低帧率跟踪综述、detect-track-pose 解耦、YOLO-Pose 对比）在全局知识库 `~/.claude/kb/30-areas/video-tracking/`，本文不重复、仅引用。

## 0. 一句话目标

在 `py/jxl` 新建子包 **`jxl.vdt`**（video-detect-track），实现一条 **detect → track →（可选）pose** 的视频处理管线：检测用 YOLO，跟踪可在 **IoU 模式**（ByteTrack，正常帧率）与 **ReID 模式**（DINOv3 嵌入 + gallery，低帧率 ≤1fps）间配置切换，pose 作为确认为人后的可选第二步（RTMPose on crop，门控）。**库核心**（纯编排，返回结构化 `Tracks`）+ **CLI**（Typer，导出 tracks JSON + 标注视频）。**Rust 生产移植**到 `next/ml` 是未来工作，本 spec 仅映射、不实现。

## 1. 范围与不在范围

**在范围（Python 原型）**
- 5 阶段正交流水线：Decoder → Detector → Tracker（IoU|ReID）→ [Pose] → Aggregator
- 双模跟踪，配置指定（`tracker = "iou" | "reid"`）
- 条件性 pose 解耦（门控 + crop 上 RTMPose）
- 视频文件批处理（MKV/MP4 → tracks JSON + 标注视频）
- 库核心可被 import（返回 `Tracks`），CLI 是薄消费者

**不在范围**
- ❌ Rust 实现（仅设计映射，见 §12）
- ❌ 实时流（RTSP/GStreamer）—— `next/ml` 现无流栈，引入成本高，留未来
- ❌ 训练 / 标注工具（jxl 已有 `label/`、`det_mine`，不重复）
- ❌ 自动帧率自适应（tracker 模式由配置显式指定，不自动切换——YAGNI）

## 2. jxl 代码库现状（复用基础）

新会话务必先读：`py/jxl/README.md`、`pyproject.toml`、`src/jxl/det/d2d.py`、`src/jxl/det/yolo/d2d_yolo.py`、`src/jxl/track/iou_tracker.py`。

- **包结构**：`src/jxl/` 下 `det/ track/ seg/ cls/ label/ io/ iqa/ sam/ yolo/ bin/`。新子包 `vdt/` 与之平级。
- **构建**：hatchling + uv，`src/` layout，`requires-python ~=3.12.0`。本地路径依赖 `jcx`（`~/cc/py/jcx`，fs/系统）与 `jvi`（`~/cc/py/jvi`，vision/图像/绘制：`Rect`、`ImageNda`、绘制）。
- **检测（成熟，复用）**：`D2dYolo`（`src/jxl/det/yolo/d2d_yolo.py`）包 ultralytics YOLO，`detect(image, persist=True)` 内部按 `D2dOpt.track` 走 `model.track` 或 `model.predict`。结果模型 `D2dObject`/`D2dResult`（`src/jxl/d2d.py`）含 normalized `Rect`、`id`、`cls`、`conf`。adapter：`det/yolo/adapter.py` `boxes_to_d2d`、`yolo/results.py` `results_list_to_d2d_result`。
- **跟踪**：`jxl.track.IouTracker`（`src/jxl/track/iou_tracker.py`）纯 Python IoU 跟踪，有 `RectObject` Protocol（`id`/`life`/`rect()`）。**无 ReID、无 ByteTrack 字面量**（ByteTrack 仅经 ultralytics `model.track` 触达）。
- **Pose**：**无**（净新增）。
- **视频 IO（碎片化，本模块要补）**：`io/video.py` 仅 `CmpVideoMaker`（拼对比片）。现有 bins 用 ffmpeg 子进程（`mkv_keyframes.py`、`ff_extract.py`）、imageio（`video_to_frames.py`）、opencv（`cv2`）。**本模块新建 `OcvDecoder`**（opencv VideoCapture，已是依赖）。
- **约定（必须遵守）**：
  - **pydantic `BaseModel` 强制**作数据模型（j-python-strict）；状态ful/不可序列化对象（如 tracker）显式避免 pydantic 并在 docstring 注明。
  - **CLI = Typer**（新标准）：`typer.Typer()` + `@app.command()` + `Annotated[...]`，参考 `src/jxl/bin/d2d_peoplenet_check.py`。entry point 在 `pyproject.toml` `[project.scripts]`。
  - **配置 = TOML**（stdlib `tomllib`），参考 `targets/*.toml` → `TargetProfile(BaseModel)` 模式。
  - **mypy strict-ish**（`strict=false` 但 `warn_return_any`/`warn_unreachable`/`no_implicit_optional`/`check_untyped_defs` 开）；`src/jxl/bin/` 外零 `Any`。第三方无 stub（cv2/ultralytics/onnxruntime）在 `[[tool.mypy.overrides]]` 列。
  - 日志 `loguru`，JSON `orjson`，`rustshed`（Rust 风 `Result`/`Err`）可见于新代码。
  - 重 ML 栈已是依赖：torch、torchvision、ultralytics(>=8.4.65, AGPL)、onnx/onnxruntime-gpu、opencv_python、imageio[ffmpeg]。

## 3. 架构 — 方案 B：正交阶段流水线

> **为何选 B（detect-first 统一）而非 A（tracker 拥有检测）**：用户首要标准是"优美的设计"。B 把检测做成**单一可替换一等阶段**，IoU 与 ReID **真正对称**（都是"对每帧检测框做关联"，同一 `Tracker.update(detections)` 接口），完美贴合 tracking-by-detection 文献范式，pose 作干净第四阶段。A 让 iou 模式把检测藏在 ultralytics 里、reid 模式摊在外面——"检测"关注点活两处，抽象泄漏。详见 kb `20260801-detect-track-pose-architecture.md`。

```
VideoDecoder ─▶ Detector ─▶ Tracker ─▶[PoseStep]─▶ Aggregator ─▶ Tracks
  (帧,ts_ms)   D2dObject[]   D2dObject[]   keypoints     Tracks      (时间线)
              (id=None)      (带 track_id)               (结构化)
              可换 YOLO/     可换 IoU/ReID
              PeopleNet/YOLOE 同一接口
```

- 每阶段 = `typing.Protocol`（结构化子类型），单一职责，可独立替换与测试。
- **Tracker 吃检测框**（不吃 image）——核心对称点。Detector 与 Tracker 完全分离，换一次检测器两种跟踪模式同时受益。
- 库核心 `pipeline.run()` 纯编排（解码器注入，无硬编码 IO）；CLI 薄消费者。

### 模块布局

```
src/jxl/vdt/
├── __init__.py
├── types.py        # 配置 + 结果模型（pydantic）：VdtConfig, FrameResult, Track, Tracks
├── decoder.py      # OcvDecoder（opencv，可配置 fps 采样）
├── detector.py     # Detector Protocol + YoloDetector（包 D2dYolo，detect 不 track）
├── tracker.py      # Tracker Protocol + IouTracker / ReidTracker
├── reid.py         # ReidEmbedder（DINOv3 ONNX/ort）+ Gallery + associate() 纯函数
├── pose.py         # PoseStep（RTMPose on crop，门控）
├── pipeline.py     # run() 编排（纯，Functional Core）
└── cli.py          # Typer CLI（bin/vdt.py 暴露 entry point）
```

## 4. 组件 — 协议、impl、数据模型

### 数据模型（`types.py`，pydantic）

```python
class D2dObject:      # 复用 jxl.det.d2d：rect, conf, cls, id
    ...               # Detector 产出时 id=None；Tracker 填 track_id（单一类型贯穿）
class Keypoints(BaseModel): pts: list[Point2d]; conf: list[float]          # COCO 17 点
class FrameResult(BaseModel):
    frame_idx: int; ts_ms: int
    objects: list[D2dObject]                # 带 track_id
    kpts: list[Keypoints | None]            # 与 objects 对齐；无 pose 则 None
class Track(BaseModel): id: int; cls: int; frames: list[FrameResult]       # 一条身份的时间线
class Tracks(BaseModel):
    src: str; fps: float; duration_ms: int
    tracks: list[Track]
    config: VdtConfig                       # 快照（可复现）
class VdtConfig(BaseModel):
    tracker: Literal["iou", "reid"]
    decode: DecodeCfg
    det: DetCfg
    tracker_cfg: IouCfg | ReidCfg
    pose: PoseCfg | None
```

### 阶段协议

```python
class Decoder(Protocol):
    def __iter__(self) -> Iterator[tuple[int, int, np.ndarray]]: ...   # (frame_idx, ts_ms, image)

class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[D2dObject]: ...        # 无 id

class Tracker(Protocol):
    def update(self, frame_idx: int, ts_ms: int, dets: list[D2dObject]) -> list[D2dObject]: ...  # 填 id
    def reset(self) -> None: ...                                       # 视频边界清状态

class PoseStep(Protocol):
    def step(self, image: np.ndarray, tracked: list[D2dObject]) -> list[Keypoints | None]: ...
```

### impl

| 阶段 | impl | 关键点 |
|---|---|---|
| Decoder | `OcvDecoder` | opencv VideoCapture；按 `DecodeCfg.fps` 抽帧；发射**源视频真实 `ts_ms`**（ReID 的 ttl/motion_radius 按秒，必须用 ts_ms 不能用采样序号） |
| Detector | `YoloDetector` | 包 `D2dYolo`，调 detect 分支（`track=False`），映射成 `D2dObject(id=None)` |
| Tracker(iou) | `IouTracker` | 复用 `jxl.track.IouTracker`（或 BoxMOT `ByteTrack`-on-detections）；忽略 `ts_ms`，按 `frame_idx` 关联 |
| Tracker(reid) | `ReidTracker` | 持 `ReidEmbedder`+`Gallery`；`update`：每检测提嵌入 → 调纯函数 `associate()` → 填 id；`reset` 清 gallery |
| ReID 嵌入 | `ReidEmbedder` | DINOv3 ViT-S/16 ONNX（ort session）；crop 预处理（resize 224/normalize）→ forward → CLS-token → L2 归一化 |
| ReID 关联 | **`associate()` 纯函数** | iap 算法（详见 §5）；Functional Core，零副作用，充分单测 |
| Pose | `RtmposeStep` | RTMPose-m ONNX；门控（详见 §6）；crop 批量 forward；坐标回映 `+= crop.xymin` |
| Aggregator | `aggregate()` 纯函数 | 逐帧 `FrameResult` → 按 track_id 聚成 `Tracks` 时间线；标记 ended track |

### 关键设计点
- **`associate()`/`aggregate()` 是纯函数核心**（设计原则 6：Functional Core, Imperative Shell）；`ReidTracker` 是持 ort session/gallery 的命令式外壳。关联算法每分支单测。
- **`D2dObject` 单一类型贯穿 detect→track**（Detector 出 `id=None`，Tracker 填 id），避免 Detection/Tracked 双类型膨胀。
- **`Tracker.reset()`** 在视频边界调用（批处理多视频防跨视频身份泄漏——对应 Rust `OnnxTracker::reset()` 已知坑）。

## 5. ReID 关联算法（`associate()` 纯函数 — 低帧率模式核心）

> 移植自 `rs/iap` 的 `reid_assoc.rs::associate()`（已验证、有完整单测集），HSV 换成 DINOv3。理论依据见 kb `20260801-lowframerate-tracking.md`（低帧率下 IoU/Kalman 必然失效，关联只能靠外观）。

```python
def associate(
    embeddings: list[np.ndarray],   # 本帧各检测的 DINOv3 嵌入（L2 归一化）
    detections: list[D2dObject],    # 本帧检测（含位置）
    gallery: Gallery,               # 既有轨迹：{id: TrackState(emb, last_pos, last_ts, hit_count)}
    ts_ms: int, cfg: ReidCfg,
) -> tuple[list[D2dObject], Gallery]:   # 带id检测 + 新gallery（纯，不 mutate 入参）
```

步骤：
1. **TTL 淘汰**：`ts_ms - last_ts > cfg.ttl_sec*1000` 的 gallery 轨迹移除（其 track 在 `Tracks` 中显式 ended）。
2. **候选对**：`(det_i, gal_j)` 同时满足 ① motion 门控（`det_i.rect` 中心距 `gal_j.last_pos` ≤ `cfg.motion_radius`，归一化坐标）② 余弦相似度 `sim(emb_i, gal_j.emb) ≥ cfg.cos`。
3. **降序贪心一对一**：候选按相似度降序，依次配对（最高 sim 先配，配过的 det/gal 不再参与）——近似 Hungarian，O(n²) 但帧内检测数小。
4. **匹配后更新 gallery**：EMA 融合嵌入 `emb = cfg.ema*new + (1-cfg.ema)*old`，更新 `last_pos`/`last_ts`/`hit_count`。
5. **未匹配检测**：开新轨迹，分配全局单调 `next_id`（**不复用**已结束 id）。

**默认参数**（iap 值 + kb 建议）：`cos=0.6`、`motion_radius=0.3`（归一化）、`ema_alpha=0.2`（kb 建议，iap 原 0.5 偏激进，生产更稳用 0.1-0.3）、`ttl_sec=600`。

**单测**（照搬 iap）：同人复用 id / 异人新建 / motion 门控拦截 / TTL 过期 / 多检测最高相似度胜出 / EMA 融合。喂合成嵌入+检测，零模型依赖。

## 6. Pose 门控（`RtmposeStep` — 解耦第二步）

> 详见 kb `20260801-detect-track-pose-architecture.md`（detect→track→conditional pose 解耦管线）。要点：别用 yolov8-pose 在 crop 上重跑（耦合 + 紧裁丢手足关键点），用纯单人回归器 RTMPose。

门控（哪些 tracked 本帧跑 pose）——`PoseStep` 内部维护 per-id 状态（`first_seen_ts`/`last_pose_ts`/上一框 aspect），**不依赖 Tracker 内部状态**：
- `class == person`（0）
- **已确认**：该 id 已被观察到 ≥ `min_hits` 帧（PoseStep 自记，避免 Tracker 暴露内部）
- **触发**（满足任一即跑）：① 周期关键帧（距上次 pose ≥ `keyframe_every` 帧）；② staleness 兜底（≥ `K_max=2*keyframe_every`）；③ aspect-ratio 跳变 `Δ>0.3`（站→坐/转身）；④ 刚跨过 min_hits（首次）；⑤ 遮挡退出（id 重新出现）
- 其余帧：复用上次 keypoints（可选：按 bbox 仿射 warp，原型可先不做）

**crop forward**：RTMPose-m ONNX；预处理 `hbb2cs(padding=1.25)` + `top_down_affine`（输入 256×192，mean/std 固定）→ SimCC 解码（x/y 各 argmax）→ 17 点 + conf。**坐标回映**：`kpt.xy += crop.xymin`。crop 过小/零面积 → `Keypoints=None`（显式 null）。

**默认**：`keyframe_every=5`、`min_hits=3`、`K_max=10`、`kpt_shape=[17,3]`。可选叠加 OneEuro 时间平滑（原型可后置）。

## 7. 模型栈与 ONNX 来源

| 槽位 | 默认模型 | 参数量 | 来源 / ONNX |
|---|---|---|---|
| 检测 | YOLO26s（`yolo26s.pt`，repo root 已有） | — | ultralytics；`D2dYolo` |
| IoU 跟踪 | ByteTrack（吃检测框） | — | `jxl.track.IouTracker` 或 BoxMOT（AGPL，与 jxl 已接受一致） |
| **ReID 嵌入** | **DINOv3 ViT-S/16**（frozen） | **~21M** | 权重 HF DINOv3 collection / [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)，`transformers≥4.56`；ONNX 经 `torch.onnx.export` 或社区 [onnx-community/convert-to-onnx](https://huggingface.co/spaces/onnx-community/convert-to-onnx)。fallback：DINOv2 small（`sefaburak/dinov2-small-onnx` 现成） |
| Pose | RTMPose-m | ~6M | MMPose / HF；usls 已实现 SimCC 解码可参照 |

> DINOv3 选型理由见 kb `20260801-lowframerate-tracking.md` §选型详解：自监督 frozen 零样本、跨域稳、Gram anchoring 修 dense 退化、判别力数量级超 HSV。ViT-S/16(21M) 与 DINOv2 small 同量级，0.5fps 下算力可忽略。

## 8. 数据流（端到端）

```python
def run(video_path: str, config: VdtConfig) -> Tracks:
    decoder  = build_decoder(video_path, config.decode)
    detector = build_detector(config.det)
    tracker  = build_tracker(config)                      # Iou | Reid
    pose     = build_pose(config.pose) if config.pose else None
    tracker.reset()
    results: list[FrameResult] = []
    for frame_idx, ts_ms, image in decoder:
        dets    = detector.detect(image)                  # id=None
        tracked = tracker.update(frame_idx, ts_ms, dets)  # 填 track_id
        kpts    = pose.step(image, tracked) if pose else [None]*len(tracked)
        results.append(FrameResult(frame_idx, ts_ms, tracked, kpts))
    return aggregate(video_path, decoder.fps, decoder.duration_ms, results, config)
```

- **管线对双模式无感**：`build_tracker(config.tracker)` 返回哪个 impl，`update` 就走哪条。
- **低帧率采样**在 Decoder：`fps=0.5` → 每 2s 抽一帧；`ts_ms` 是源视频真实时间（ReID 阈值按秒/源像素）。

## 9. 错误处理（No Silent Degradation / fail-fast）

严格遵守 j 编码规范——资源缺失或降级一律立即失败，禁止静默回退（无 `FALLBACK` 标注不回退替代模型）：

| 场景 | 处理 |
|---|---|
| 视频打不开 / 0 帧 | `raise`，中止 |
| 权重缺失 / ONNX 加载失败 / ort EP 不可用 | `raise`（显式路径，不回退别的模型） |
| 配置未知 `tracker` / 非法阈值 | pydantic `Literal` + validator，加载即 fail-fast |
| `DecodeCfg.fps` 抽帧得 0 帧 | `raise`（配置错误） |
| 检测空帧（无人） | 正常，`dets=[]` |
| ReID 无匹配 | 开新 track_id（正常）；TTL 到期 → track 显式 `ended` |
| Pose crop 退化 / track 失联 >K_max | `keypoints=None` 显式 null（不静默填零点） |

异常类型具体化（`DecodeError`/`ModelLoadError`/`ReidError`/`PoseError`），不裸 `Exception`。

## 10. 测试

Functional Core（`associate`/`aggregate`/`pipeline.run`）→ 充分单测：

- **`associate()`**：照搬 iap 单测集（§5），合成嵌入+检测+gallery，零模型依赖。
- **`aggregate()`**：合成 `FrameResult[]`，断言时间线/id 连续性/ended/pose None。
- **`IouTracker`**：合成检测序列（固定/位移/漏检），断言跨帧 id。
- **`PoseStep` 门控**：合成 tracked+时序，断言哪些 frame/track 被 posed。
- **`OcvDecoder`**：合成短视频，断言帧数/fps 采样/`ts_ms`。
- **`YoloDetector`/`ReidEmbedder`/`RtmposeStep`**：集成测试（真实模型 + 小 fixture），不进单测；单测用 Protocol fake 注入。
- **`pipeline.run()`**：冒烟（tiny 视频+模型）+ 无 pose 路径 + 双模式各一条。
- **`VdtConfig`**：validator 快照（非法值 fail-fast）。

mypy strict（`src/jxl/bin/` 外零 `Any`）。注入点全 Protocol 化。

## 11. 配置与 CLI

**TOML**（`experiments/vdt-*.toml`，沿用 jxl `targets/*.toml` 风格）：
```toml
tracker = "reid"                          # 或 "iou"
[decode]  fps = 0.5                       # 低帧率；iou 模式可设 25
[det]     model = "yolo26s.pt"; conf = 0.4; iou = 0.5; classes = [0]
[tracker_cfg.reid]  model = "dinov3-vits16.onnx"; cos = 0.6; motion_radius = 0.3; ema = 0.2; ttl_sec = 600
# [tracker_cfg.iou]  max_age = 30; min_hits = 3; iou = 0.3
[pose]    enabled = true; model = "rtmpose-m.onnx"; kpt_shape = [17,3]; keyframe_every = 5; min_hits = 3
```

**CLI**（Typer，参考 `d2d_peoplenet_check.py`；entry point `vdt` 加进 `pyproject.toml [project.scripts]`，指向 `jxl.bin.vdt:app` 或 `jxl.vdt.cli:app`）：
```bash
vdt run video.mkv --config experiments/vdt-person.toml \
     --tracker reid --out-tracks tracks.json --out-video annotated.mp4
vdt info    # 打印可用模型/配置
```
标注视频用 opencv VideoWriter + `jvi` 绘制；tracks JSON 用 orjson 序列化 `Tracks`。

## 12. Rust 移植映射（实现暂缓，仅映射）

| Python (`jxl.vdt`) | Rust (`next/ml`) | 备注 |
|---|---|---|
| `OcvDecoder` | generalize `peoplenet-cli/video_decoder.rs`（ffmpeg-next） | 现唯一 decode 循环，去 peoplenet 耦合 |
| `YoloDetector` | `ml-vision::OnnxDetector`（Detector trait） | 已有 |
| `Tracker.update(dets)` | **`ml-vision::Tracker` trait 重构为吃 detections** | **B 驱动修正**：OnnxTracker 拆 Detector+Tracker，更 SRP |
| `IouTracker` | `usls::ByteTracker`（已有） | |
| `ReidTracker`+`ReidEmbedder` | 新 impl（DINOv3 via `ort` + Gallery） | 新建 |
| `associate()` | 纯 Rust fn（1:1 移植） | |
| `RtmposeStep` | `ml-vision::PoseEstimator` trait + usls RTMPose adapter | usls 已有 |
| `pipeline.run()` | **缺失的"视频推理组装层" crate**（ml-vision 之上，不依赖 ml-runtime） | 代码已点名缺口 |
| 配置 | serde + workspace 配置 | |

> Rust 端关键动作：`Tracker` trait 从"吃 image"改"吃 detections"——把 Python 原型验证的更干净形状回写 Rust（与 kb `20260801-detect-track-pose-architecture.md` 一致）。

## 13. 实现分期（设计是一体，实现可分）

虽三个能力一起设计，实现建议按依赖顺序：

- **P1 — 骨架 + 检测 + IoU 跟踪（MVP）**：`types`+`decoder`+`detector`+`IouTracker`+`aggregate`+`pipeline.run`+CLI（无 pose、iou 模式）。让模块存在并跑通正常帧率视频。风险最低，复用最多。
- **P2 — 条件性 Pose**：加 `RtmposeStep` + 门控；`--pose` 开关。
- **P3 — 低帧率 ReID 模式**：`ReidEmbedder`(DINOv3 ONNX) + `associate()` + `ReidTracker`；`--tracker reid`。最大新件、最高风险，Python 原型在此最有价值（快速试 DINOv3）。

每阶段独立可跑、可测、可 commit。

## 14. 已定决策日志

- 范围 = 多模式跟踪(D) + pose 解耦；三个一起设计（不拆 spec）。
- 部署形态 = 库核心 + CLI 批处理（Q2-D）；流暂缓。
- 实现 = **仅 Python 原型**（Rust 暂缓）。
- tracker 模式 = 配置指定（不自动切换）。
- 架构 = **方案 B**（detect-first 正交流水线），理由=优美（检测一等阶段、IoU/ReID 对称）。
- 模型栈 = YOLO26 / DINOv3 ViT-S/16 / RTMPose-m / ByteTrack-on-detections。
- 数据模型 = 复用 `D2dObject` 单类型（id 由 Tracker 填）。
- 错误处理 = No Silent Degradation / fail-fast。

## 15. 深度背景引用（全局 kb，新会话按需读）

- `~/.claude/kb/30-areas/video-tracking/20260801-lowframerate-tracking.md` — 低帧率跟踪综述、IoU 崩塌数学、ReID 选型详解（DINOv3>DINOv2>SOLIDER>CLIP-ReID）、iap HSV 评估
- `~/.claude/kb/30-areas/video-tracking/20260801-detect-track-pose-architecture.md` — detect→track→pose 解耦设计、门控策略表、RTMPose 选型
- `~/.claude/kb/30-areas/video-tracking/20260801-dayn9t-tracking-inventory.md` — dayn9t 现有跟踪实现盘点（iap ReID 算法来源）
- `~/rs/iap/iap-master/src/monitor/reid_assoc.rs` + `src/detector/reid.rs` — `associate()`/HandCraftedReid 原始实现（算法移植源）
