"""vdt 检测器实现：``YoloDetector`` 包装 ``D2dYolo``（detect 分支，不 track）。

实现 spec §4 Detector 行 + §7。``Detector`` 协议见 :mod:`jxl.vdt.types`：

- 构造时 lazy import 重依赖（``D2dYolo``/``D2dOpt``/``ultralytics``），避免
  ``import jxl.vdt.detector`` 拉入 ultralytics。
- 权重缺失/加载失败 → ``ModelLoadError``（No Silent Degradation：显式路径，
  不回退别的模型）。
- ``detect`` 走 ``D2dOpt.track=False`` predict 分支，``D2dObject.id`` 恒为 ``0``
  哨兵（由 Tracker 填 ``track_id >= 1``）。
- 按 ``DetCfg.classes`` 过滤；空集合 = 不过滤。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from jvi.image.image_nda import ImageNda
from jxl.det.d2d import D2dObject
from jxl.vdt.types import DetCfg, ModelLoadError

if TYPE_CHECKING:
    from jxl.det.d2d import D2dResult


class _DetectorLike(Protocol):
    """``D2dYolo`` 的结构化窄接口（ISP）：仅依赖 ``detect(image)->D2dResult``。"""

    def detect(self, image: ImageNda) -> "D2dResult": ...


class YoloDetector:
    """``Detector`` 协议的具体实现：包装 :class:`jxl.det.yolo.d2d_yolo.D2dYolo`。

    有状态（持有 YOLO 模型 session 与类别白名单），仅程序内构造；非可序列化
    （``D2dYolo`` 内含 ``ultralytics.YOLO``，不走 pydantic）。

    ``detect`` 返回 ``id=0`` 哨兵的 ``D2dObject`` 列表，``track_id`` 由后续
    ``Tracker`` 阶段填入。
    """

    def __init__(self, cfg: DetCfg) -> None:
        """构造检测器。

        Args:
            cfg: 检测配置（model/conf/iou/classes/device/input_shape）。

        Raises:
            ModelLoadError: 权重路径不存在或 YOLO 加载失败（不回退替代模型）。
        """
        # lazy import：重依赖（ultralytics/torch），避免模块 import 时拉入。
        from jxl.det.d2d import D2dOpt
        from jxl.det.yolo.d2d_yolo import D2dYolo

        model_path = Path(cfg.model)
        opt = D2dOpt(
            input_shape=cfg.input_shape,
            conf_thr=cfg.conf,
            iou_thr=cfg.iou,
            track=False,
        )
        try:
            self._det: _DetectorLike = D2dYolo(model_path, opt, cfg.device)
        except (FileNotFoundError, OSError, ModuleNotFoundError, ImportError) as ex:
            raise ModelLoadError(
                f"加载 YOLO 检测权重失败: {model_path}（{type(ex).__name__}: {ex}）"
            ) from ex
        except Exception as ex:  # ultralytics 内部抛类型不一，统一收口为 ModelLoadError。
            raise ModelLoadError(
                f"YOLO 权重加载异常: {model_path}（{type(ex).__name__}: {ex}）"
            ) from ex

        self._classes: set[int] = set(cfg.classes)

    def detect(self, image: np.ndarray) -> list[D2dObject]:
        """对一帧 BGR ndarray 执行检测，返回 ``id=0`` 的 ``D2dObject`` 列表。

        Args:
            image: BGR ``np.ndarray``（与 :class:`jvi.image.image_nda.ImageNda`
                ``data()`` 同约定，D2dYolo 内部直接消费 BGR）。

        Returns:
            过滤 ``DetCfg.classes`` 后的检测结果；空集合表示不过滤。每个
            ``D2dObject.id == 0``（哨兵，未被关联）。
        """
        img = ImageNda(data=image)
        result = self._det.detect(img)
        objs = result.objects
        if self._classes:
            objs = [o for o in objs if o.cls in self._classes]
        return objs


# ---------------------------------------------------------------------------
# 单测（pytest 自动发现）。
# YoloDetector 依赖 ultralytics + 真实权重 + 可选 GPU → 属集成测试（spec §10）。
# 真实模型 smoke 用守卫跳过；契约/过滤逻辑用 Protocol fake 零模型依赖验证。
# ---------------------------------------------------------------------------


class _FakeD2dYolo:
    """``D2dYolo`` 的 Protocol fake（鸭子类型），仅用于单测。

    模拟 ``track=False`` 分支：返回固定 ``D2dResult``，``id`` 恒为 0。
    """

    def __init__(self, factory: Callable[[], list[D2dObject]]) -> None:
        self._factory = factory

    def detect(self, image: ImageNda) -> "D2dResult":  # noqa: ARG002
        """fake detect：忽略 image，返回注入构造的对象。"""
        from jxl.det.d2d import D2dResult

        return D2dResult(objects=self._factory())


def _make_obj(cls: int, conf: float = 0.9, oid: int = 0) -> D2dObject:
    """构造测试用 D2dObject（全归一化坐标）。"""
    from jvi.geo.rectangle import Rect

    return D2dObject(id=oid, cls=cls, conf=conf, rect=Rect.from_ltrb_list([0.1, 0.1, 0.4, 0.4]))


def test_detect_filters_by_classes() -> None:
    """空 classes=不过滤；指定 classes 只保留命中类别。"""
    det = YoloDetector.__new__(YoloDetector)
    det._det = _FakeD2dYolo(lambda: [_make_obj(0), _make_obj(2), _make_obj(5)])

    det._classes = set()
    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 3
    assert all(o.id == 0 for o in out)

    det._classes = {0}
    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].cls == 0


def test_detect_returns_empty_when_no_match() -> None:
    """classes 过滤后无匹配返回空列表（正常空帧）。"""
    det = YoloDetector.__new__(YoloDetector)
    det._det = _FakeD2dYolo(lambda: [_make_obj(1), _make_obj(2)])
    det._classes = {0}
    assert det.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


def test_detect_sentinel_id_zero() -> None:
    """detect 分支 id 恒为 0（哨兵），与 iap/Rust associate() 约定一致。"""
    det = YoloDetector.__new__(YoloDetector)
    det._det = _FakeD2dYolo(lambda: [_make_obj(0, oid=0), _make_obj(0, oid=0)])
    det._classes = set()
    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert all(o.id == 0 for o in out)


def test_detect_handles_empty_result() -> None:
    """D2dResult.objects 为空时 detect 返回空列表（无人场景）。"""
    det = YoloDetector.__new__(YoloDetector)
    det._det = _FakeD2dYolo(lambda: [])
    det._classes = set()
    assert det.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


def test_init_raises_on_missing_weight(tmp_path: Path) -> None:
    """权重不存在 → ModelLoadError（No Silent Degradation）。"""
    cfg = DetCfg(model=str(tmp_path / "nonexistent.pt"))
    try:
        YoloDetector(cfg)
    except ModelLoadError as ex:
        assert "nonexistent.pt" in str(ex)
    else:
        raise AssertionError("期望 ModelLoadError 未抛出")


# --- 集成 smoke（真实模型 + 视频，守卫跳过） -----------------------------

def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def test_smoke_yolo_detector_on_real_video() -> None:
    """集成 smoke：抽一帧 → YoloDetector → 断言 id=0 / cls 在白名单内。

    需 repo root 权重 + quyang-street.mp4 + 可选 GPU；缺一即跳过（spec §10
    将 YoloDetector 归为集成项）。
    """
    import pytest

    repo_root = Path(__file__).resolve().parents[3]
    weight = repo_root / "yolo26n.pt"
    video = Path.home() / "cc/py/jvi/assets/video/quyang-street.mp4"
    if not weight.exists() or not video.exists() or not _has_cuda():
        pytest.skip("缺 yolo26n.pt / 视频fixture / CUDA，跳过集成 smoke")

    import cv2

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None

    det = YoloDetector(DetCfg(model=str(weight), classes=[0]))
    objs = det.detect(frame)
    assert isinstance(objs, list)
    assert all(o.id == 0 for o in objs)
    assert all(o.cls == 0 for o in objs)
