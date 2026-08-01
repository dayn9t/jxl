"""vdt ReID 跟踪器实现：``ReidTracker``（spec §4 Tracker(reid) 行）。

``Tracker`` 协议（:mod:`jxl.vdt.types`）的命令式外壳：持 ``Embedder`` + ``Gallery``，
``update`` 对每个检测提外观嵌入 → 调纯函数 :func:`jxl.vdt.reid_assoc.associate` →
填 ``track_id`` → 用返回的新 gallery 替换内部状态。

设计要点（spec §3/§4/§5）：

- **方案 B 对称**：与 :class:`jxl.vdt.tracker.IouTracker` 同接口（``update`` 吃检测框 +
  当前帧）。``image`` 透传——IoU 忽略，ReID 用 image crop 提嵌入（提外观做关联，低帧率
  下 IoU/Kalman 必然失效，spec §5 引 kb `20260801-lowframerate-tracking.md`）。
- **Functional Core / Imperative Shell**（设计原则 6）：关联算法（TTL 淘汰 + 运动门控 +
  余弦阈值 + 贪心一对一 + EMA 融合）封装在纯函数 :func:`associate`（零副作用、有独立
  单测集）；本类只负责"提嵌入 → 调 associate → 换 gallery"的薄编排。
- **id=0 哨兵**：零面积 crop（退化检测）→ 提嵌入失败 → 零向量 → :func:`associate`
  识别为零范数（移植自 iap `reid_assoc.rs`，与 ``embedding_valid`` 同源）→ emit ``id=0``
  哨兵（不匹配、不新建、不消耗 next_id），由 aggregate 过滤。
- **Embedder 注入**（ISP/可测试性，设计原则 2/10）：生产由 ``build_tracker`` 构造
  :class:`jxl.vdt.reid.ReidEmbedder`（DINOv2 ONNX）注入；测试传 fake embedder（受控
  嵌入），零模型依赖。
- **No Silent Degradation**（spec §9）：embedder 抛错向上传播（不吞错、不静默回退）；
  退化 crop 走显式零向量→哨兵路径（显式 null 语义，不静默填假数据）。

兄弟模块依赖（签名已钉死，并行期 ``unresolved-import`` 可暂忽略，主控集成 mypy）：

- :mod:`jxl.vdt.reid_assoc`：``associate(embeddings, detections, gallery, ts_ms, cfg)``
  → ``(list[D2dObject], Gallery)``；``Gallery(tracks, next_id)``；``Embedder`` 协议。
- :mod:`jxl.vdt.reid`：``ReidEmbedder``（生产 embedder，集成测试用）。
"""

from __future__ import annotations

import numpy as np
from jvi.geo.rectangle import Rect
from jvi.geo.size2d import Size

from jxl.det.d2d import D2dObject
from jxl.vdt._geom import pixel_box
from jxl.vdt.reid_assoc import Embedder, Gallery, associate
from jxl.vdt.types import ReidCfg

# ---------------------------------------------------------------------------
# Functional Core：_crop 纯函数（设计原则 6；裁剪原语单一数据源见 _geom.pixel_box）
# ---------------------------------------------------------------------------


def _crop(
    image: np.ndarray, rect: Rect, img_w: int, img_h: int
) -> np.ndarray | None:
    """归一化 ``rect`` → crop ndarray；零面积 → None。

    裁剪原语（clip/零面积判据）复用 :func:`jxl.vdt._geom.pixel_box`（单一数据源，
    与 pose 共用）；ReID 仅需 crop 本身，不需坐标回映偏移。
    """
    box = pixel_box(rect, img_w, img_h)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Imperative Shell：``ReidTracker``（持 Embedder + Gallery）
# ---------------------------------------------------------------------------


class ReidTracker:
    """``Tracker`` 协议实现（ReID 模式）：持 ``Embedder`` + ``Gallery``。

    有状态（``_gallery`` 每帧替换为 :func:`associate` 返回的新实例——纯函数不 mutate
    入参，shell 持久化其返回值），**非可序列化**，仅程序内构造——状态ful 的 gallery /
    嵌入维度刻意避免 pydantic（设计原则 5：可变状态显式声明且最小化）。

    与 :class:`IouTracker` 完全对称（spec §3 方案 B）：同 ``update(frame_idx, ts_ms,
    image, dets)`` 接口；IoU 忽略 ``image``/``ts_ms``（按 ``frame_idx`` 关联），ReID 用
    ``image`` crop 提嵌入、按 ``ts_ms`` 做 TTL（spec §4/§5）。``frame_idx`` 在 reid 模式
    忽略——仅保留以满足协议对称（换一次检测器两种跟踪模式同时受益）。
    """

    def __init__(self, cfg: ReidCfg, embedder: Embedder) -> None:
        """构造 ReID 跟踪器。

        Args:
            cfg: ReID 跟踪配置（``model``/``cos``/``motion_radius``/``ema``/``ttl_sec``）。
                ``cfg.model`` 由生产端 ``ReidEmbedder`` 构造时使用（本类不直接加载模型——
                embedder 注入）；``cos``/``motion_radius``/``ema``/``ttl_sec`` 透传给
                :func:`associate`。
            embedder: 嵌入提取器（生产传 ``ReidEmbedder``，测试传 fake——ISP/可测试性）。
        """
        self._cfg: ReidCfg = cfg
        self._embedder: Embedder = embedder
        self._gallery: Gallery = Gallery(tracks={}, next_id=1)
        self._ended_ids: set[int] = set()
        self._dim: int | None = None
        """嵌入维度（模型内禀，跨帧/跨 reset 缓存）。首帧首个有效嵌入发现后锁定；
        零面积 crop 的占位零向量用此维度。None = 尚未见有效嵌入。"""

    @property
    def ended_ids(self) -> set[int]:
        """本视频被 TTL 过期淘汰的 track_id（供 aggregate 标 ``ended``，spec §9）。"""
        return self._ended_ids

    def reset(self) -> None:
        """视频边界清 gallery + ended_ids（批处理多视频防跨视频身份泄漏，spec §4）。

        清空轨迹、``next_id`` 归 1——新视频从头分配，不继承上一视频的身份。
        ``_dim`` 跨 reset 保留（嵌入维度是模型内禀属性，非视频状态；保留避免新视频首帧
        重新探测）。
        """
        self._gallery = Gallery(tracks={}, next_id=1)
        self._ended_ids = set()

    def update(
        self,
        frame_idx: int,
        ts_ms: int,
        image: np.ndarray,
        dets: list[D2dObject],
    ) -> list[D2dObject]:
        """提嵌入 → 关联 → 填 id → 更新 gallery。

        ``frame_idx`` 在 reid 模式忽略（仅 ``ts_ms`` 参与 TTL/关联，spec §4/§5）；保留
        签名满足协议对称性。``image`` 用于 crop 提嵌入（与 IoU 忽略 image 形成对称）。

        Args:
            frame_idx: 帧序（忽略；reid 按 ``ts_ms`` 关联）。
            ts_ms: 源视频真实时间戳 ms（TTL 淘汰按秒：``ts_ms - last_ts > ttl_sec*1000``）。
            image: 当前 BGR 帧（提嵌入用）。
            dets: 本帧检测（``id=0`` 哨兵，归一化 rect）。

        Returns:
            与 ``dets`` 同序的 ``D2dObject`` 列表，``track_id >= 1``（已关联）或 ``0``
            （零面积 crop / 提取失败哨兵，由 aggregate 过滤）。
        """
        img_h, img_w = image.shape[:2]
        crops = [_crop(image, d.rect, img_w, img_h) for d in dets]
        embeddings = self._embed_all(crops)
        old_ids = set(self._gallery.tracks)
        tracked, new_gallery = associate(
            embeddings, dets, self._gallery, ts_ms, self._cfg
        )
        # TTL 淘汰：旧 gallery 有、新 gallery 无的 id（associate 内部按 ttl_sec 过滤）。
        self._ended_ids |= old_ids - set(new_gallery.tracks)
        self._gallery = new_gallery
        return tracked

    def _embed_all(self, crops: list[np.ndarray | None]) -> list[np.ndarray]:
        """对每个 crop 提嵌入；``None``（零面积）→ 零向量（id=0 哨兵，由 associate 识别）。

        两遍处理：先提有效 crop 发现/复用嵌入维度（``_dim`` 跨帧缓存），再用零向量填
        ``None`` 位——避免 None 位与有效嵌入维度不一致（``np.zeros`` 需具体维度，且
        不同维度向量无法做余弦）。零范数向量与 :func:`associate` 的 ``embedding_valid``
        契约对齐（移植自 iap，零范数 = 提取失败 → ``id=0`` 哨兵）。
        """
        out: list[np.ndarray | None] = [None] * len(crops)
        for i, crop in enumerate(crops):
            if crop is not None:
                emb = self._embedder.embed(crop)
                if self._dim is None:
                    self._dim = int(emb.shape[0])
                out[i] = emb
        dim = self._dim if self._dim is not None else 1
        return [
            emb if emb is not None else np.zeros(dim, dtype=np.float32)
            for emb in out
        ]


# ---------------------------------------------------------------------------
# 单测（pytest 自动发现；spec §10 ReID 项）。
# 自包含：合成 image + fake embedder（按 crop 像素均值映射到 one-hot 嵌入，模拟同人/异人），
# 零真实模型依赖（reid_tracker 已硬依赖 reid_assoc，无需运行时 importorskip）。
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


def _det(
    x: float, y: float, w: float = 0.2, h: float = 0.4, oid: int = 0
) -> D2dObject:
    """构造测试用 ``D2dObject``（归一化 rect，``id=oid``；默认 id=0 哨兵）。"""
    return D2dObject(id=oid, cls=0, conf=1.0, rect=Rect.new(x, y, w, h))


def _paint(image: np.ndarray, rect: Rect, value: int) -> np.ndarray:
    """在 ``image`` 的归一化 rect 像素区域填常量 ``value``（全通道），返回 ``image``。

    像素框计算与 :func:`_crop` 同源（``absolutize``+``round``+clip），用于在测试图上
    为不同区域涂不同值，驱动 fake embedder 产出区分性嵌入。
    """
    img_h, img_w = image.shape[:2]
    px = rect.absolutize(Size.new(img_w, img_h)).round()
    lt, rb = px.ltrb()
    x0, y0 = max(0, int(lt.x)), max(0, int(lt.y))
    x1, y1 = min(img_w, int(rb.x)), min(img_h, int(rb.y))
    image[y0:y1, x0:x1] = value
    return image


class _FakeEmbedder:
    """测试 Embedder：crop 像素均值 → one-hot 嵌入（同人/异人模拟）。

    - 同均值（``% dim`` 同）→ 同 one-hot 单位向量（``cos=1``，同人）；
    - 不同均值（``% dim`` 不同）→ 不同 one-hot（正交，``cos=0``，异人）。

    满足 ``Embedder`` 协议（``embed(crop)->ndarray``）；``dim`` 仅为自描述（协议不要求）。
    测试通过 :func:`_paint` 给区域涂不同常量值来控制嵌入。
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self._basis = np.eye(dim, dtype=np.float32)  # 行 i = one-hot i（单位向量）

    def embed(self, crop: np.ndarray) -> np.ndarray:
        idx = int(crop.mean()) % self.dim
        # 就地构造 one-hot（避免 ndarray 索引返回 Any）。
        v = np.zeros(self.dim, dtype=np.float32)
        v[idx] = 1.0
        return v


_IMG_SIZE = 100
"""测试图边长（正方形，100x100）。"""


def _blank() -> np.ndarray:
    """全零测试图（BGR, 100x100x3）。"""
    return np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)


def test_crop_clips_and_slices() -> None:
    """``_crop``：归一化 rect → 像素 clip → ndarray；与 pose._pixel_box 同源行为。"""
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    # rect (0.1,0.2,0.5,0.6) → 像素 (10,20,w=50,h=60) → image[20:80, 10:60]
    crop = _crop(image, Rect.new(0.1, 0.2, 0.5, 0.6), 100, 100)
    assert crop is not None
    assert crop.shape == (60, 50, 3)


def test_crop_zero_area_returns_none() -> None:
    """``_crop`` 零面积 rect（w=0）→ None。"""
    assert _crop(_blank(), Rect.new(0.1, 0.1, 0.0, 0.4), 100, 100) is None




def test_same_person_reuses_id_across_frames() -> None:
    """同人跨帧复用 id：同位置同纹理 det → 相同嵌入 → associate 复用同一 track_id。"""
    trk = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    image = np.full((_IMG_SIZE, _IMG_SIZE, 3), 10, dtype=np.uint8)  # 全图常量 → crop mean=10
    d = _det(0.1, 0.1)
    f1 = trk.update(0, 0, image, [d])
    f2 = trk.update(1, 1000, image, [d])  # ts_ms +1s

    assert f1[0].id >= 1
    assert f2[0].id == f1[0].id  # 复用（cos=1, 同位置过 motion 门控）


def test_distinct_persons_get_distinct_ids() -> None:
    """异人新建：不同位置 + 不同纹理 → 正交嵌入 → 各自新 id（不混淆）。"""
    trk = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    image = _blank()
    a, b = _det(0.1, 0.1), _det(0.7, 0.1)  # 中心距 ≈ 0.6 > motion_radius=0.3
    _paint(image, a.rect, 1)  # → one-hot 1
    _paint(image, b.rect, 2)  # → one-hot 2（正交）

    out = trk.update(0, 0, image, [a, b])
    assert out[0].id >= 1 and out[1].id >= 1
    assert out[0].id != out[1].id


def test_distinct_embedding_at_same_position_gets_new_id() -> None:
    """跨帧同人位置但外观变（正交嵌入）→ 不复用旧 id（cos 门控拦截，motion 放行）。"""
    trk = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    d = _det(0.1, 0.1)

    img1 = _blank()
    _paint(img1, d.rect, 1)  # 帧0：外观 A
    f1 = trk.update(0, 0, img1, [d])

    img2 = _blank()
    _paint(img2, d.rect, 5)  # 帧1：同位置、外观 B（正交，cos=0 < 0.6）
    f2 = trk.update(1, 1000, img2, [d])

    assert f1[0].id >= 1
    assert f2[0].id != f1[0].id  # cos 门控拦下，开新 id


def test_ttl_expiry_assigns_new_id() -> None:
    """TTL 过期：ts_ms 跨度 > ``ttl_sec*1000`` → 旧轨迹被淘汰，再现分配新 id。"""
    trk = ReidTracker(ReidCfg(model="<fake>", ttl_sec=1), _FakeEmbedder())  # 1 秒 TTL
    image = np.full((_IMG_SIZE, _IMG_SIZE, 3), 10, dtype=np.uint8)
    d = _det(0.1, 0.1)

    f1 = trk.update(0, 0, image, [d])  # id=1, last_ts=0
    oid = f1[0].id
    f2 = trk.update(1, 2000, image, [d])  # 间隔 2s > ttl 1s → 旧轨迹淘汰 → 新 id

    assert f2[0].id != oid
    assert f2[0].id >= 1


def test_ttl_marks_ended() -> None:
    """TTL 淘汰的轨迹进 ``ended_ids``（供 aggregate 标 ``Track.ended``，spec §9/§10）。"""
    trk = ReidTracker(ReidCfg(model="<fake>", ttl_sec=1), _FakeEmbedder())
    image = np.full((_IMG_SIZE, _IMG_SIZE, 3), 10, dtype=np.uint8)
    d = _det(0.1, 0.1)

    out = trk.update(0, 0, image, [d])  # id=1
    oid = out[0].id
    assert trk.ended_ids == set()  # 尚无淘汰
    # 人离开（空检测）超过 ttl → gallery 淘汰 id=oid
    trk.update(1, 1500, image, [])
    assert oid in trk.ended_ids


def test_reset_clears_gallery_no_cross_video_leak() -> None:
    """reset 清 gallery：``tracks`` 空、``next_id`` 归 1（新视频不继承旧轨迹）。"""
    trk = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    image = np.full((_IMG_SIZE, _IMG_SIZE, 3), 10, dtype=np.uint8)
    d = _det(0.1, 0.1)
    trk.update(0, 0, image, [d])
    assert len(trk._gallery.tracks) == 1  # 有轨迹

    trk.reset()
    assert trk._gallery.tracks == {}  # 清空
    assert trk._gallery.next_id == 1  # next_id 归 1

    f2 = trk.update(0, 0, image, [d])  # 新视频首帧
    assert len(trk._gallery.tracks) == 1  # 新轨迹，不继承
    assert f2[0].id >= 1


def test_zero_area_crop_emits_id0_sentinel() -> None:
    """零面积 crop（w=0）→ 提嵌入失败 → 零向量 → ``id=0`` 哨兵（不新建、不消耗 next_id）。"""
    trk = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    image = np.full((_IMG_SIZE, _IMG_SIZE, 3), 10, dtype=np.uint8)
    zero = D2dObject(id=0, cls=0, conf=1.0, rect=Rect.new(0.1, 0.1, 0.0, 0.4))  # w=0

    out = trk.update(0, 0, image, [zero])
    assert out[0].id == 0  # 哨兵
    assert trk._gallery.next_id == 1  # 未消耗 next_id


def test_frame_idx_ignored_reid_uses_ts_ms() -> None:
    """``frame_idx`` 在 reid 模式忽略：相同 ts_ms + image + dets，frame_idx 任意变 → 同结果。"""
    trk1 = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    trk2 = ReidTracker(ReidCfg(model="<fake>"), _FakeEmbedder())
    image = np.full((_IMG_SIZE, _IMG_SIZE, 3), 10, dtype=np.uint8)
    d = _det(0.1, 0.1)

    # frame_idx 截然不同，ts_ms 相同 → 应产出相同 id（reid 按 ts_ms 关联）。
    f1 = trk1.update(0, 0, image, [d])
    f2 = trk2.update(999, 0, image, [d])
    assert f1[0].id == f2[0].id  # frame_idx 不影响


# -- _embed_all 维度管理（不依赖 associate，验证 shell 的 dim 发现/复用） ------------


class _RecordingEmbedder:
    """记录每次 embed 输入的 fake embedder；返回固定维度 one-hot（首维=1）。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[np.ndarray] = []

    def embed(self, crop: np.ndarray) -> np.ndarray:
        self.calls.append(crop)
        v = np.zeros(self.dim, dtype=np.float32)
        v[0] = 1.0
        return v


def test_embed_all_discovers_dim_and_fills_zeros() -> None:
    """``_embed_all``：有效 crop 提嵌入并发现 dim；None 位用同维度零向量填充。"""
    emb = _RecordingEmbedder(dim=4)
    trk = ReidTracker(ReidCfg(model="<fake>"), emb)
    valid = np.ones((10, 10, 3), dtype=np.uint8)
    # 有效 crop 提嵌入；None 位（零面积）→ 零向量（dim 与有效嵌入一致）。
    out = trk._embed_all([valid, None, valid])
    assert len(emb.calls) == 2  # 仅有效位调 embedder
    assert out[0].shape == (4,) and out[2].shape == (4,)
    assert np.array_equal(out[1], np.zeros(4, dtype=np.float32))  # 同 dim 零向量
    assert trk._dim == 4  # 维度已锁定


def test_embed_all_dim_cached_across_frames() -> None:
    """``_dim`` 跨帧缓存：首帧锁定后，次帧即使首检测为零面积也用缓存维度。"""
    emb = _RecordingEmbedder(dim=4)
    trk = ReidTracker(ReidCfg(model="<fake>"), emb)
    valid = np.ones((10, 10, 3), dtype=np.uint8)
    trk._embed_all([valid])  # 锁定 dim=4
    out = trk._embed_all([None])  # 次帧零面积 → 用缓存 dim
    assert out[0].shape == (4,)
    assert np.array_equal(out[0], np.zeros(4, dtype=np.float32))


# -- 真实 DINOv2 集成 smoke（本地有权重 + 兄弟模块就绪时验证连通） ------------------


def test_integration_real_dinov2_same_person_reuses_id() -> None:
    """真实 DINOv2 ViT-S/14 smoke：跨帧同纹理同位置 crop → 嵌入一致 → 不丢 id。

    依赖 ``onnxruntime`` + ``jxl.vdt.reid``（ReidEmbedder）+ ``jxl.vdt.reid_assoc`` 三者
    就绪，且本地存在 ``dinov2_vits14.onnx``；缺任一则优雅跳过。
    """
    pytest.importorskip("onnxruntime")
    pytest.importorskip("jxl.vdt.reid")
    onnx = Path("/home/jiang/cc/py/jxl/dinov2_vits14.onnx")
    if not onnx.is_file():
        pytest.skip("无 dinov2_vits14.onnx（集成 smoke 跳过）")

    from jxl.vdt.reid import ReidEmbedder  # 兄弟模块；签名 (model_path) — 集成时按实际对齐

    cfg = ReidCfg(model=str(onnx))
    emb = ReidEmbedder(str(onnx))
    trk = ReidTracker(cfg, emb)
    # 固定纹理（可复现）：同一 crop 同一输入 → DINOv2 输出一致 → cos=1.0 >= thr → 复用 id。
    rng = np.random.RandomState(0)
    texture = rng.randint(0, 256, (_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
    d = _det(0.1, 0.1, 0.4, 0.6)

    f1 = trk.update(0, 0, texture, [d])
    f2 = trk.update(1, 1000, texture, [d])  # 同一 crop → 同一嵌入 → 复用

    assert f1[0].id >= 1
    assert f2[0].id == f1[0].id
