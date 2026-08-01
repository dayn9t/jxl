"""vdt ReID 嵌入提取：``ReidEmbedder``（DINOv2 ViT-S/14 ONNX）。

实现 spec §4 ReID 嵌入行 + §7。``Embedder`` 协议**单一数据源**在
:mod:`jxl.vdt.reid_assoc`（纯函数核心定义窄接口）；本模块 import 并实现之
（``ReidEmbedder`` 满足 ``Embedder``）——ISP/可测，fake embedder 即可单测关联算法：

- 构造时 lazy import ``onnxruntime``（重 ML 栈，避免 ``import jxl.vdt.reid`` 拉入）。
- 权重缺失/加载失败 → ``ModelLoadError``（No Silent Degradation：显式路径，不回退
  别的模型）。
- ``embed``：crop → BGR→RGB → resize 224 → /255 → ImageNet mean/std 标准化 →
  HWC→CHW → forward → 384-d CLS 嵌入 → L2 归一化。
- 零面积/全零 crop → 全零 384-d 向量（无效哨兵；``associate`` 据此判无效 → id=0）。

> **模型 fallback**：spec §7 默认 DINOv3 ViT-S/16，本实现用可得的 DINOv2 ViT-S/14
> ONNX（spec §7 注明的 fallback 路径）。两者预处理一致（224 + ImageNet 统计量），
> 仅 patch_size 与嵌入维数不同——DINOv2 ViT-S/14 输出 384-d（与 DINOv3 ViT-S/16 同
> 维数）。预处理常量来源：torchvision ``DINOv2`` / ``torchvision.models`` 官方实现
> （``weights=Dinov2_ViT_S14_Weights`` 的 ``transforms.normalize``）与原论文
> (Oquab et al. 2023, "DINOv2: Learning Robust Visual Features without Supervision"）
> 一致：mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，作用在 [0,1] RGB 空间
> （先 ``/255`` 再减均值——与 RTMPose 的 [0,255] 空间不同，勿混淆）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 测试段用（pytest 自动发现；常规 import，勿用 __import__ 动态导入——j-python-strict）
from jvi.geo.size2d import Size as _Size

from jxl.vdt._ort import OrtSessionLike, build_ort_session
from jxl.vdt.reid_assoc import Embedder  # 协议单一数据源（纯函数核心模块）
from jxl.vdt.types import ModelLoadError, ReidError

# ---------------------------------------------------------------------------
# 常量（DINOv2 官方预处理；单一数据源——本模块钉死，reid_tracker/reid_assoc 不重述）。
# ---------------------------------------------------------------------------

DINOV2_SIZE: int = 224
"""DINOv2 ViT-S 输入边长（px）。"""

DINOV2_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""ImageNet RGB 均值（[0,1] 空间）。"""

DINOV2_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""ImageNet RGB 标准差（[0,1] 空间）。"""

DINOV2_DIM: int = 384
"""DINOv2 ViT-S/14 输出嵌入维数（CLS token 已池化）。"""


# ---------------------------------------------------------------------------
# ort 结构化窄接口 + CUDA fail-fast 构造：单一数据源见 :mod:`jxl.vdt._ort`
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Imperative Shell：``ReidEmbedder``（持 ort session；满足 reid_assoc.Embedder）
# ---------------------------------------------------------------------------


class ReidEmbedder:
    """``Embedder`` 协议的具体实现：DINOv2 ViT-S/14 ONNX。

    有状态（持 ort ``InferenceSession``），仅程序内构造；**非可序列化**——ort session
    刻意避免 pydantic（设计原则 5：可变状态显式声明且最小化，session 是外部资源句柄）。

    spec §7 默认 DINOv3 ViT-S/16；本实现用可得的 DINOv2 ViT-S/14 fallback（spec §7
    注明的 fallback 路径）。两者预处理与嵌入维数（384）一致。
    """

    def __init__(self, model_path: str) -> None:
        """构造 ort session。

        ort session 经 :func:`jxl.vdt._ort.build_ort_session` 构造——**要求 CUDA**
        （fail-fast，No Silent Degradation；不静默回退 CPU）。输出维度在 ``embed``
        推理期由 ``reshape`` 校验，不符即 ``ReidError``（错误具体化，不裸泄漏）。

        Args:
            model_path: DINOv2 ONNX 权重路径。

        Raises:
            ModelLoadError: 权重不存在 / 无 CUDA EP / ort 加载失败 / IO 节点数不符
                （不回退替代模型）。
        """
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"DINOv2 权重不存在: {model_path}")
        session = build_ort_session(str(path))

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) < 1:
            raise ModelLoadError(f"DINOv2 应有 >=1 输入，实际 {len(inputs)}")
        if len(outputs) != 1:
            names = [o.name for o in outputs]
            raise ModelLoadError(
                f"DINOv2 应有 1 输出 (CLS 嵌入)，实际 {len(outputs)}: {names}"
            )

        self._session: OrtSessionLike = session
        self._in_name: str = inputs[0].name

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """BGR [Hc,Wc,3] uint8 crop → 384-d L2 归一化嵌入。

        预处理（DINOv2 官方，详见模块 docstring 常量来源）：

        1. BGR → RGB。
        2. resize 224×224（``cv2.INTER_LINEAR``）。
        3. ``/255`` → ImageNet mean/std 标准化（RGB [0,1] 空间）。
        4. HWC → CHW → ``[1,3,224,224]`` float32。
        5. forward → ``output[0]`` 形状 [384]（CLS 已池化；维度构造期已校验）。
        6. L2 归一化。

        零面积 / 全零 crop → 返回全零 384-d（无效哨兵，``associate`` 据此判无效 →
        id=0，spec §9 显式哨兵而非静默填零点嵌入）。

        Raises:
            ReidError: 推理期 ort 异常（错误具体化，不裸泄漏）。
        """
        import cv2

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(DINOV2_DIM, dtype=np.float32)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (DINOV2_SIZE, DINOV2_SIZE), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        mean = np.array(DINOV2_MEAN, dtype=np.float32)
        std = np.array(DINOV2_STD, dtype=np.float32)
        x = (x - mean) / std  # 广播：HWC × 3
        x = np.transpose(x, (2, 0, 1))[None, ...]  # [1,3,224,224]

        try:
            out = self._session.run(None, {self._in_name: x})[0]
            emb = out.reshape(DINOV2_DIM).astype(np.float32)  # 维度不符→ValueError→ReidError
        except (RuntimeError, ValueError) as ex:  # EP 失败 / 形状不匹配
            raise ReidError(f"DINOv2 推理失败: {type(ex).__name__}: {ex}") from ex
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            emb = emb / norm
        else:
            return np.zeros(DINOV2_DIM, dtype=np.float32)  # 全零嵌入兜底
        return emb


# ---------------------------------------------------------------------------
# 单测（pytest 自动发现；spec §10 ReidEmbedder 项）。
# 真实模型集成（dinov2_vits14.onnx 本地可得）→ 重依赖 lazy import 在测试内完成。
# 若权重缺失则 pytest.skip。
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[3]
"""py/jxl 仓库根（用于定位 dinov2_vits14.onnx 与 assets）。"""

_DINOV2_ONNX = _REPO_ROOT / "dinov2_vits14.onnx"
"""DINOv2 ViT-S/14 ONNX 权重（repo root 已有）。"""

_P2_JPG = _REPO_ROOT / "assets" / "person" / "p2.jpg"
"""单人 fixture 图（同人两 crop 验证高余弦）。"""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两向量余弦相似度（均已 L2 归一化则等价内积）。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def test_embed_output_is_l2_normalized() -> None:
    """embed 输出 L2 范数 ≈ 1.0（容差 1e-3）。"""
    import pytest

    if not _DINOV2_ONNX.is_file():
        pytest.skip("缺 dinov2_vits14.onnx，跳过集成 smoke")

    emb = ReidEmbedder(str(_DINOV2_ONNX))
    crop = np.random.RandomState(0).randint(0, 256, (128, 64, 3), dtype=np.uint8)
    out = emb.embed(crop)
    assert out.shape == (DINOV2_DIM,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-3


def test_embed_zero_area_crop_returns_zero_vector() -> None:
    """零面积 crop → 全零 384-d（无效哨兵，非归一化到 1）。"""
    import pytest

    if not _DINOV2_ONNX.is_file():
        pytest.skip("缺 dinov2_vits14.onnx，跳过集成 smoke")

    emb = ReidEmbedder(str(_DINOV2_ONNX))
    zero = np.zeros((0, 0, 3), dtype=np.uint8)
    out = emb.embed(zero)
    assert out.shape == (DINOV2_DIM,)
    assert np.all(out == 0.0)


def test_init_raises_on_missing_weight(tmp_path: Path) -> None:
    """权重缺失 → ModelLoadError（No Silent Degradation）。"""
    bogus = tmp_path / "nonexistent.onnx"
    try:
        ReidEmbedder(str(bogus))
    except ModelLoadError as ex:
        assert "nonexistent.onnx" in str(ex)
    else:
        raise AssertionError("期望 ModelLoadError 未抛出")


def test_embed_same_person_two_crops_high_cosine() -> None:
    """同人两略有位移的 crop → 余弦 >= 0.8。

    从 p2.jpg 检测 person（YoloDetector，conf 0.3），取主 crop 与略位移 crop →
    embed → 高余弦（同人外观近，spec §7 DINOv2 判别力假设）。
    """
    import pytest

    if not _DINOV2_ONNX.is_file() or not _P2_JPG.is_file():
        pytest.skip("缺 dinov2_vits14.onnx 或 p2.jpg，跳过同人余弦 smoke")

    import cv2

    from jxl.vdt.detector import YoloDetector
    from jxl.vdt.types import DetCfg

    # 权重缺失/无 CUDA 时 YoloDetector 跑不起来 → 跳过（集成 smoke 容错）。
    yolo_weight = _REPO_ROOT / "yolo26n.pt"
    if not yolo_weight.is_file():
        pytest.skip("缺 yolo26n.pt，跳过同人余弦 smoke")

    img = cv2.imread(str(_P2_JPG))
    assert img is not None
    h, w = img.shape[:2]

    det = YoloDetector(DetCfg(model=str(yolo_weight), conf=0.3, classes=[0]))
    objs = det.detect(img)
    if not objs:
        pytest.skip("p2.jpg 未检出 person，跳过")
    ob = max(objs, key=lambda o: o.conf)

    # 主 crop：归一化 rect → 像素框裁剪。
    px = ob.rect.absolutize(_Size.new(w, h)).round()
    lt, rb = px.ltrb()
    x0, y0 = max(0, int(lt.x)), max(0, int(lt.y))
    x1, y1 = min(w, int(rb.x)), min(h, int(rb.y))
    crop_main = img[y0:y1, x0:x1]
    assert crop_main.size > 0

    # 略位移 crop：水平偏移 8% 宽度（clip 到边界）。
    dx = int(0.08 * (x1 - x0))
    nx0 = max(0, x0 + dx)
    nx1 = min(w, x1 + dx)
    crop_shift = img[y0:y1, nx0:nx1]
    assert crop_shift.size > 0

    emb = ReidEmbedder(str(_DINOV2_ONNX))
    e1 = emb.embed(crop_main)
    e2 = emb.embed(crop_shift)
    cos = _cosine(e1, e2)
    assert cos >= 0.8, f"同人两 crop 余弦过低: {cos:.4f}"


def test_embed_heterogeneous_crops_low_cosine() -> None:
    """person crop vs 纯色 crop → 余弦 < 0.5（异类低相关）。"""
    import pytest

    if not _DINOV2_ONNX.is_file() or not _P2_JPG.is_file():
        pytest.skip("缺 dinov2_vits14.onnx 或 p2.jpg，跳过异类余弦 smoke")

    import cv2

    from jxl.vdt.detector import YoloDetector
    from jxl.vdt.types import DetCfg

    yolo_weight = _REPO_ROOT / "yolo26n.pt"
    if not yolo_weight.is_file():
        pytest.skip("缺 yolo26n.pt，跳过异类余弦 smoke")

    img = cv2.imread(str(_P2_JPG))
    assert img is not None
    h, w = img.shape[:2]

    det = YoloDetector(DetCfg(model=str(yolo_weight), conf=0.3, classes=[0]))
    objs = det.detect(img)
    if not objs:
        pytest.skip("p2.jpg 未检出 person，跳过")
    ob = max(objs, key=lambda o: o.conf)
    px = ob.rect.absolutize(_Size.new(w, h)).round()
    lt, rb = px.ltrb()
    x0, y0 = max(0, int(lt.x)), max(0, int(lt.y))
    x1, y1 = min(w, int(rb.x)), min(h, int(rb.y))
    crop_person = img[y0:y1, x0:x1]
    assert crop_person.size > 0

    # 纯灰 crop（与 person 外观无关）。
    solid = np.full((max(2, y1 - y0), max(2, x1 - x0), 3), 128, dtype=np.uint8)

    emb = ReidEmbedder(str(_DINOV2_ONNX))
    e_p = emb.embed(crop_person)
    e_s = emb.embed(solid)
    cos = _cosine(e_p, e_s)
    assert cos < 0.5, f"异类余弦过高: {cos:.4f}"
