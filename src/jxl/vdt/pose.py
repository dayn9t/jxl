"""vdt 条件性 Pose 实现：``RtmposeStep``（spec §6）。

``PoseStep`` 协议（:mod:`jxl.vdt.types`）的命令式外壳：门控决策 → crop →
RTMPose-m ONNX 批量 forward → SimCC 解码 → 坐标回映归一化 → 复用上次关键点。

设计要点（spec §6）：

- **门控解耦**：``RtmposeStep`` 仅消费 ``PoseGate.step(id,cls,frame_idx,aspect)``
  的布尔决策，不依赖 ``Tracker`` 内部状态。门控启发式（min_hits 确认 / keyframe_every
  周期 / aspect 跳变 / K_max 兜底）封装在 :mod:`jxl.vdt.pose_gate`，本模块只负责"按
  决策跑或不跑 + 复用缓存"。
- **zero-forward 优化**：本帧无任何 ``decide=True`` 目标 → 不调 ``_forward``（省算力，
  解耦收益）。多个 decide 目标 → 批量预处理 → 拼 batch → **单次** ``_forward`` →
  分发解码（O(N) forward 收敛为 O(1)）。
- **坐标回映**：``simcc_decode`` 产出的 ``kps_crop`` 在 crop 自身像素系（基于 crop 的
  center/scale）；全帧像素 = crop 坐标 + crop 左上角像素 (x0,y0)；归一化 = /img_w,/img_h。
- **No Silent Degradation**（spec §9）：权重缺失 / ort 加载失败 / 输入或输出数不符 →
  ``ModelLoadError``，不回退别的模型。crop 退化 / decode 失败 → ``Keypoints=None``
  显式 null（不静默填零点）。

兄弟模块依赖（签名已钉死，并行期 ``unresolved-import`` 可暂忽略，主控集成 mypy）：

- :mod:`jxl.vdt.rtmpose_proc`：``preprocess_crop(crop)->(tensor,center,scale)``、
  ``simcc_decode(simcc_x,simcc_y,center,scale)->Keypoints|None``。
- :mod:`jxl.vdt.pose_gate`：``PoseGate(cfg)``，``gate.step(id,cls,frame_idx,aspect)->bool``、
  ``gate.reset()``。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from jvi.geo.point2d import Point
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size
from jxl.det.d2d import D2dObject
from jxl.vdt.types import Keypoints, ModelLoadError, PoseCfg

# ---------------------------------------------------------------------------
# ort 结构化窄接口（ISP：仅依赖本模块用到的方法；避免引用被 mypy ignore 的
# onnxruntime，也避免 ``Any``——j-python-strict）。
# ---------------------------------------------------------------------------


class _OrtNodeLike(Protocol):
    """ort 输入/输出节点窄接口（仅用 ``name``）。"""

    name: str


class _OrtSessionLike(Protocol):
    """``ort.InferenceSession`` 的结构化窄接口（仅用 ``run``/``get_inputs``/``get_outputs``）。"""

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...

    def get_inputs(self) -> Sequence[_OrtNodeLike]: ...

    def get_outputs(self) -> Sequence[_OrtNodeLike]: ...


# ---------------------------------------------------------------------------
# Functional Core：纯函数（设计原则 6；可独立单测）
# ---------------------------------------------------------------------------


def _pixel_box(
    rect: Rect, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """归一化 ``Rect`` → 裁剪到边界的像素 int 框 ``(x0, y0, x1, y1)``；零面积 → None。

    ``D2dObject.rect`` 恒归一化（来自 detector）。``absolutize`` 内部已 ``round``，
    再 ``.round()`` 幂等确保整数；随后裁剪到 ``[0,img_w]×[0,img_h]``（detector 可能
    越界，clip 兜住）。
    """
    px = rect.absolutize(Size.new(img_w, img_h)).round()
    lt, rb = px.ltrb()
    x0 = max(0, int(lt.x))
    y0 = max(0, int(lt.y))
    x1 = min(img_w, int(rb.x))
    y1 = min(img_h, int(rb.y))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1


def _crop_rect(
    image: np.ndarray, rect: Rect, img_w: int, img_h: int
) -> tuple[np.ndarray, int, int] | None:
    """裁剪 ``rect`` 对应的 crop，返回 ``(crop, x0_px, y0_px)``；零面积 → None。

    左上角像素 ``(x0_px, y0_px)`` 是坐标回映的偏移量，与 crop 同源于一次裁剪——单一
    数据源（设计原则 8），避免调用处重算裁剪逻辑。
    """
    box = _pixel_box(rect, img_w, img_h)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1], x0, y0


def _remap(
    kps_crop: Keypoints | None, x0: int, y0: int, img_w: int, img_h: int
) -> Keypoints | None:
    """crop 像素系关键点 → 全帧归一化（``+=crop 左上角`` 再 ``/img_w,/img_h``）。

    decode 失败（``None``）→ ``None``（spec §9 显式 null，不静默填零点）。
    """
    if kps_crop is None:
        return None
    pts = [
        Point(x=(p.x + x0) / img_w, y=(p.y + y0) / img_h) for p in kps_crop.pts
    ]
    return Keypoints(pts=pts, conf=list(kps_crop.conf))


# ---------------------------------------------------------------------------
# Imperative Shell：``RtmposeStep``（持 ort session / gate / 缓存）
# ---------------------------------------------------------------------------


class RtmposeStep:
    """``PoseStep`` 协议实现：门控 + RTMPose-m ONNX on crop + SimCC 解码 + 坐标回映。

    有状态（持 ort ``InferenceSession``、``PoseGate``、``_frame_idx`` 帧序代理、
    ``_last_kpts`` per-id 复用缓存）；**非可序列化**，仅程序内构造——状态ful/ort
    session 刻意避免 pydantic（设计原则 5：可变状态显式声明且最小化）。

    ``_frame_idx`` 作 frame_idx 代理：``run_pipeline`` 每帧调一次 ``step``，内部自增
    （``reset`` 归 -1）；门控的"距上次 pose 帧数"以此为时间本。
    """

    def __init__(self, cfg: PoseCfg) -> None:
        """构造 ort session 与门控。

        lazy import ``onnxruntime``（重 ML 栈，避免模块 import 时拉入）。``providers``
        按 spec §6 显式配置 CUDA 优先 / CPU 兜底（spec §9 仅禁"回退别的模型"，不禁
        EP 优先级表）。

        Args:
            cfg: pose 配置（model/kpt_shape/keyframe_every/min_hits）。

        Raises:
            ModelLoadError: 权重缺失 / ort 加载失败 / 输入或输出数不符。
        """
        import onnxruntime as ort  # heavy ML; lazy

        model_path = Path(cfg.model)
        if not model_path.is_file():
            raise ModelLoadError(f"RTMPose 权重不存在: {cfg.model}")
        # providers 顺序按 spec §6 显式配置（CUDA 优先，CPU 兜底）。
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as ex:  # ort 抛类型不一，统一收口为 ModelLoadError。
            raise ModelLoadError(
                f"RTMPose 加载失败 ({cfg.model}): {type(ex).__name__}: {ex}"
            ) from ex

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) < 1:
            raise ModelLoadError(f"RTMPose 应有 >=1 输入，实际 {len(inputs)}")
        if len(outputs) != 2:
            names = [o.name for o in outputs]
            raise ModelLoadError(
                f"RTMPose 应有 2 输出 (simcc_x, simcc_y)，实际 {len(outputs)}: {names}"
            )

        from jxl.vdt.pose_gate import PoseGate  # 兄弟模块；并行期 unresolved 可暂忽略

        self._session: _OrtSessionLike = session
        self._in_name: str = inputs[0].name
        self._cfg: PoseCfg = cfg
        self._gate = PoseGate(cfg)
        self._frame_idx: int = -1
        self._last_kpts: dict[int, Keypoints | None] = {}

    # -- 协议方法 ----------------------------------------------------------

    def step(
        self, image: np.ndarray, tracked: list[D2dObject]
    ) -> list[Keypoints | None]:
        """对 tracked 决策门控 + 必要时跑 pose；返回与 tracked 同序的 list。

        ``id==0`` 哨兵位恒 ``None``（不经门控）。``decide=True`` 跑 pose 并缓存；
        ``decide=False`` 复用缓存（无缓存 → ``None``）。本帧无任何 decide → 不调
        ``_forward``（zero-forward 优化）。
        """
        self._frame_idx += 1
        frame_idx = self._frame_idx
        img_h, img_w = image.shape[:2]

        results: list[Keypoints | None] = [None] * len(tracked)
        pending: list[tuple[int, np.ndarray, int, int]] = []  # (pos, crop, x0, y0)
        for pos, ob in enumerate(tracked):
            if ob.id == 0:
                continue  # 哨兵：results[pos] 保持 None，不经门控
            aspect = ob.rect.aspect_ratio()
            decide = self._gate.step(ob.id, ob.cls, frame_idx, aspect)
            if not decide:
                results[pos] = self._last_kpts.get(ob.id)  # 复用（无缓存→None）
                continue
            cropped = _crop_rect(image, ob.rect, img_w, img_h)
            if cropped is None:
                self._last_kpts[ob.id] = None  # crop 退化：显式缓存 None
                continue
            crop, x0, y0 = cropped
            pending.append((pos, crop, x0, y0))

        if pending:
            decoded = self._forward_batch(pending, img_w, img_h)
            for (pos, _crop, _x0, _y0), kpts in zip(pending, decoded, strict=True):
                self._last_kpts[tracked[pos].id] = kpts
                results[pos] = kpts
        return results

    def reset(self) -> None:
        """视频边界：``gate.reset`` + ``frame_idx`` 归 -1 + 清复用缓存。

        批处理多视频时在边界调用，防跨视频身份/缓存泄漏（对应 ``IouTracker.reset``）。
        """
        self._gate.reset()
        self._frame_idx = -1
        self._last_kpts.clear()

    # -- 内部 --------------------------------------------------------------

    def _forward_batch(
        self,
        pending: list[tuple[int, np.ndarray, int, int]],
        img_w: int,
        img_h: int,
    ) -> list[Keypoints | None]:
        """批量预处理 → 单次 ``_forward`` → 分发解码 + 坐标回映。

        lazy import ``jxl.vdt.rtmpose_proc``（兄弟模块；并行期 unresolved 可暂忽略，
        主控集成 mypy）。``preprocess_crop`` 返回 ``(tensor, center, scale)``，约定
        tensor 形如 ``[1,3,H,W]``（ort-ready），squeeze 掉 singleton batch 后拼
        ``[N,3,H,W]``；兼容直接给 ``[3,H,W]``。
        """
        from jxl.vdt.rtmpose_proc import preprocess_crop, simcc_decode

        tensors: list[np.ndarray] = []
        metas: list[tuple[Point, Point, int, int]] = []  # (center, scale, x0, y0)
        for _pos, crop, x0, y0 in pending:
            tensor, center, scale = preprocess_crop(crop)
            arr = np.asarray(tensor)
            if arr.ndim == 4:
                arr = arr[0]  # drop singleton batch
            tensors.append(arr)
            metas.append((center, scale, x0, y0))

        batch = np.stack(tensors, axis=0)
        simcc_x, simcc_y = self._forward(batch)  # [N,17,W*2], [N,17,H*2]

        out: list[Keypoints | None] = []
        for i, (center, scale, x0, y0) in enumerate(metas):
            kps_crop = simcc_decode(simcc_x[i], simcc_y[i], center, scale)
            out.append(_remap(kps_crop, x0, y0, img_w, img_h))
        return out

    def _forward(self, batch_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ort 单次推理（批量），返回 ``(simcc_x, simcc_y)`` 形如 ``[N,17,W*2]``。

        不 squeeze batch——批量优化要求 ``_forward`` 接受 ``N>=1`` 并返回批量输出；
        per-crop 切片由 ``_forward_batch`` 索引完成（spec §6 单 crop squeeze 草图被
        批量优化覆盖）。``run(None, ...)`` 取全部输出（构造时已校验恰为 2 个）。
        """
        outputs = self._session.run(None, {self._in_name: batch_tensor})
        return outputs[0], outputs[1]


# ---------------------------------------------------------------------------
# 单测（pytest 自动发现；spec §10 PoseStep 门控项）。
# 自包含：合成 image/tracked + 脚本化门控 + 合成 simcc，零真实模型、零真实兄弟模块。
# ---------------------------------------------------------------------------

import sys
import types  # noqa: E402  (test 段，lazy 注册 fake 兄弟模块)
from collections.abc import Callable  # noqa: E402

import pytest  # noqa: E402


def _det(
    x: float, y: float, w: float = 0.4, h: float = 0.6, oid: int = 1
) -> D2dObject:
    """构造测试用 ``D2dObject``（归一化 rect，``id=oid``；默认 id=1 非 哨兵）。"""
    return D2dObject(id=oid, cls=0, conf=1.0, rect=Rect.new(x, y, w, h))


class _ScriptedGate:
    """脚本化门控：按给定 ``decisions`` 序列依次返回；记录调用供断言。

    替代尚未就绪的 ``PoseGate``，隔离验证 ``RtmposeStep`` 的门控**中继**逻辑（何时跑
    forward / 何时复用缓存），不依赖门控启发式本身（后者属 pose_gate 单测）。
    """

    def __init__(self, decisions: list[bool]) -> None:
        self._decisions: list[bool] = list(decisions)
        self.calls: list[tuple[int, int, int, float]] = []

    def step(self, oid: int, cls: int, frame_idx: int, aspect: float) -> bool:
        self.calls.append((oid, cls, frame_idx, aspect))
        return self._decisions.pop(0)

    def reset(self) -> None:
        self.calls.clear()


def _make_step(gate: object) -> RtmposeStep:
    """绕过 ``__init__``（避免 ort session）构造 ``RtmposeStep``，注入 fake gate。"""
    s = RtmposeStep.__new__(RtmposeStep)
    # _forward 被 monkeypatch，session 不被触达；仅占位以满足属性存在性。
    s._session = None  # type: ignore[assignment]
    s._in_name = "in"
    s._cfg = PoseCfg(model="<fake>", kpt_shape=(17, 3))
    s._gate = gate  # type: ignore[assignment]
    s._frame_idx = -1
    s._last_kpts = {}
    return s


def _install_fake_rtmpose_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    """注册 fake ``jxl.vdt.rtmpose_proc``：``preprocess_crop`` 给零张量占位；
    ``simcc_decode`` 按 (simcc_x, simcc_y) 各 keypoint 的 argmax 还原 crop 像素点。"""

    mod = types.ModuleType("jxl.vdt.rtmpose_proc")

    def preprocess_crop(crop: np.ndarray) -> tuple[np.ndarray, Point, Point]:
        tensor = np.zeros((1, 3, 8, 8), dtype=np.float32)
        center = Point(x=0.0, y=0.0)  # 占位（fake forward/decode 不依赖其值）
        scale = Point(x=1.0, y=1.0)
        return tensor, center, scale

    def simcc_decode(
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        center: Point,
        scale: Point,
    ) -> Keypoints | None:
        k = simcc_x.shape[0]
        pts = [Point(x=float(np.argmax(simcc_x[i])), y=float(np.argmax(simcc_y[i])))
               for i in range(k)]
        return Keypoints(pts=pts, conf=[1.0] * k)

    setattr(mod, "preprocess_crop", preprocess_crop)
    setattr(mod, "simcc_decode", simcc_decode)
    monkeypatch.setitem(sys.modules, "jxl.vdt.rtmpose_proc", mod)


def _spike_forward(
    n_kpt: int = 1, spike: tuple[int, int] = (2, 3), simcc_w: int = 8, simcc_h: int = 8
) -> Callable[[RtmposeStep, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """构造 fake ``_forward``：在每 batch/crop 的 (spike_x, spike_y) 处放正向尖峰。"""

    sx, sy = spike

    def fake_forward(self: RtmposeStep, batch_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = batch_tensor.shape[0]
        simcc_x = np.full((n, n_kpt, simcc_w), -1.0, dtype=np.float32)
        simcc_y = np.full((n, n_kpt, simcc_h), -1.0, dtype=np.float32)
        simcc_x[:, :, sx] = 1.0
        simcc_y[:, :, sy] = 1.0
        return simcc_x, simcc_y

    return fake_forward


# -- Functional Core 纯函数测试 -------------------------------------------


def test_pixel_box_clips_to_bounds_and_rounds() -> None:
    """归一化 rect → 像素 int 框，裁剪到边界（detector 越界被 clip）。"""
    # 100x100 图，rect (0.1,0.2,0.5,0.6) → 像素 (10,20,50,60)。
    box = _pixel_box(Rect.new(0.1, 0.2, 0.5, 0.6), 100, 100)
    assert box == (10, 20, 60, 80)


def test_pixel_box_clips_overflow() -> None:
    """越界 rect（部分超出画面）被裁剪到 [0,w]×[0,h]，不产生负坐标/超限。"""
    box = _pixel_box(Rect.new(-0.1, -0.1, 0.5, 0.5), 100, 100)
    assert box == (0, 0, 40, 40)


def test_pixel_box_zero_area_returns_none() -> None:
    """零面积 rect（w=0）→ None。"""
    assert _pixel_box(Rect.new(0.1, 0.1, 0.0, 0.5), 100, 100) is None


def test_crop_rect_returns_crop_and_offset() -> None:
    """``_crop_rect`` 返回 crop 切片 + 左上角偏移（坐标回映所需）。"""
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    out = _crop_rect(image, Rect.new(0.1, 0.2, 0.5, 0.6), 100, 100)
    assert out is not None
    crop, x0, y0 = out
    assert x0 == 10 and y0 == 20
    assert crop.shape == (60, 50, 3)  # [y0:y1, x0:x1] → (80-20, 60-10)


# -- step 门控中继测试 ----------------------------------------------------


def test_gating_only_runs_forward_when_decide_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """decide=True 跑 forward 并缓存；decide=False 复用缓存，不跑 forward。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([True, False, False, True])
    step = _make_step(gate)
    monkeypatch.setattr(RtmposeStep, "_forward", _spike_forward())

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    d = _det(0.1, 0.2)
    f0 = step.step(image, [d])
    f1 = step.step(image, [d])
    f2 = step.step(image, [d])
    f3 = step.step(image, [d])

    assert f0[0] is not None  # decide=True → 跑 forward
    assert f1[0] is not None and f1[0] == f0[0]  # decide=False → 复用
    assert f2[0] == f0[0]
    assert f3[0] is not None  # decide=True → 重跑
    # 门控每个非哨兵目标每帧调一次（共 4 次）
    assert len(gate.calls) == 4


def test_reuse_returns_none_when_no_prior_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """decide=False 且无任何先前缓存 → None（首帧即 not decide）。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([False])
    step = _make_step(gate)
    monkeypatch.setattr(RtmposeStep, "_forward", _spike_forward())

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = step.step(image, [_det(0.1, 0.2)])
    assert out == [None]


def test_id0_sentinel_skipped_no_gate_no_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """id=0 哨兵位恒 None：不门控、不 forward。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([True])  # 仅给 id=1 一张票
    step = _make_step(gate)

    calls: list[int] = []  # 记录每次调用的 batch size
    spike = _spike_forward()

    def fwd(self: RtmposeStep, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(b.shape[0])
        return spike(self, b)

    monkeypatch.setattr(RtmposeStep, "_forward", fwd)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = step.step(image, [_det(0.1, 0.2, oid=0), _det(0.1, 0.2, oid=1)])

    assert out[0] is None  # id=0 哨兵
    assert out[1] is not None  # id=1 跑了 pose
    assert len(gate.calls) == 1  # 仅 id=1 被门控
    assert gate.calls[0][0] == 1
    assert calls == [1]  # 仅 id=1 跑了一次 forward


def test_zero_forward_when_no_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """本帧无任何 decide=True → 不调 _forward（zero-forward 优化）。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([False, False])
    step = _make_step(gate)

    calls: list[int] = []

    def fwd(self: RtmposeStep, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(b.shape[0])
        return b, b

    monkeypatch.setattr(RtmposeStep, "_forward", fwd)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    step.step(image, [_det(0.1, 0.2, oid=1), _det(0.5, 0.2, oid=2)])
    assert calls == []


def test_batch_forward_single_call_for_multiple_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两 decide=True 目标同帧 → _forward 恰调用一次（批量优化）。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([True, True])
    step = _make_step(gate)

    calls: list[int] = []  # 每次调用的 batch size
    spike = _spike_forward()

    def fwd(self: RtmposeStep, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(b.shape[0])
        return spike(self, b)

    monkeypatch.setattr(RtmposeStep, "_forward", fwd)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = step.step(image, [_det(0.1, 0.2, oid=1), _det(0.5, 0.2, oid=2)])

    assert calls == [2]  # 单次批量，batch 维 = 2
    assert out[0] is not None and out[1] is not None
    assert out[0] != out[1]  # 不同 id 不同缓存项


# -- 坐标回映测试 ---------------------------------------------------------


def test_remap_normalizes_into_unit_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """fake forward 在 crop 坐标 (2,3) 放尖峰 → 回映后落在 [0,1] 归一化范围。

    image 100x100，rect=(0.1,0.2,0.5,0.6) → crop 左上角 (10,20)。
    尖峰 (2,3) + (10,20) → 全帧像素 (12,23) → 归一化 (0.12, 0.23)。
    """
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([True])
    step = _make_step(gate)
    monkeypatch.setattr(RtmposeStep, "_forward", _spike_forward(spike=(2, 3)))

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = step.step(image, [_det(0.1, 0.2, 0.5, 0.6)])
    assert out[0] is not None
    pt = out[0].pts[0]
    assert pt.x == pytest.approx(0.12)
    assert pt.y == pytest.approx(0.23)
    assert 0.0 <= pt.x <= 1.0 and 0.0 <= pt.y <= 1.0


def test_crop_zero_area_returns_none_and_caches_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """crop 零面积 → 该位 None；缓存 None 使后续 not-decide 帧也 None。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([True, False])  # 帧0 crop 退化；帧1 not decide 复用缓存
    step = _make_step(gate)
    monkeypatch.setattr(RtmposeStep, "_forward", _spike_forward())

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    zero_rect = _det(0.1, 0.2, 0.0, 0.6)  # w=0 → 零面积
    f0 = step.step(image, [zero_rect])
    f1 = step.step(image, [zero_rect])

    assert f0 == [None]
    assert f1 == [None]  # 复用缓存 None


# -- reset / __init__ 测试 ------------------------------------------------


def test_reset_clears_cache_and_frame_idx(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset() 清缓存：之前缓存的 kpts 不再被复用；frame_idx 归位。"""
    _install_fake_rtmpose_proc(monkeypatch)
    gate = _ScriptedGate([True, False, False])  # reset 后第二帧 not decide
    step = _make_step(gate)
    monkeypatch.setattr(RtmposeStep, "_forward", _spike_forward())

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    d = _det(0.1, 0.2)
    f0 = step.step(image, [d])
    assert f0[0] is not None

    step.reset()
    assert step._last_kpts == {}  # 缓存清空
    assert step._frame_idx == -1

    # reset 后首帧 decide=False → 无缓存 → None（而非复用旧值）
    f1 = step.step(image, [d])
    assert f1 == [None]


def test_init_raises_on_missing_model(tmp_path: Path) -> None:
    """权重文件不存在 → ModelLoadError（No Silent Degradation：fail-fast，不静默）。"""
    bogus = tmp_path / "nope.onnx"
    with pytest.raises(ModelLoadError, match="不存在"):
        RtmposeStep(PoseCfg(model=str(bogus)))


# -- 集成 smoke（真实模型；本地无权重则跳过） ------------------------------


def test_smoke_real_model_optional(tmp_path: Path) -> None:
    """真实 RTMPose onnx smoke：_crop_rect + preprocess + 真实 session 不崩。

    本地通常无 rtmpose*.onnx → importorskip/skip 优雅跳过； CI 有权重时验证连通。
    """
    pytest.importorskip("onnxruntime")
    pytest.importorskip("jxl.vdt.rtmpose_proc")
    candidates = list(Path(tmp_path).glob("rtmpose*.onnx")) + list(
        Path("/home/jiang/cc/py/jxl").glob("rtmpose*.onnx")
    )
    if not candidates:
        pytest.skip("无 rtmpose onnx 权重（集成 smoke 跳过）")

    step = RtmposeStep(PoseCfg(model=str(candidates[0])))
    image = np.zeros((256, 192, 3), dtype=np.uint8)
    out = step.step(image, [_det(0.1, 0.1, 0.8, 0.8)])
    assert len(out) == 1  # 不崩即通过
