"""OSNet Embedder 单测——自 iapx ``tests/test_osnet.py`` 迁移（P2-T7）。

迁移改动（余者原样）：

- import 路径：``iapx.pipeline.osnet`` → ``jxl.vdt.reid_osnet``；
  ``iapx.vendor.osnet`` → ``jxl.vdt.vendored.osnet``。
- 类名：``OsnetEmbedding`` → ``OsnetEmbedder``（jxl 协议实现者命名）。
- 批量接口：iapx ``embed(crops)`` → ``embed_batch(crops)``（协议适配，
  见 ``reid_osnet.py`` 模块 docstring）；协议单 crop ``embed`` 另测
  （退化 crop 哨兵语义 = 与 iapx raise 语义的唯一分叉点，各自锁定）。
- ``OSNET_WEIGHTS`` 枚举断言对齐 4-tag 现状（iapx 原测试停在 3-tag，
  ft-v21 追加后未同步——迁移时修正）；权重盘点改 jxl 重依赖门惯例
  （缺权重盘 ``pytest.skip``，同 ``reid.py`` 对 dinov2 权重的处理）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jxl.vdt.reid_assoc import Embedder
from jxl.vdt.reid_osnet import (
    OSNET_DEFAULT_TAG,
    OSNET_DIM,
    OSNET_INPUT_HW,
    OSNET_WEIGHTS,
    WEIGHTS_DIR,
    OsnetEmbedder,
)
from jxl.vdt.vendored.osnet import osnet_x0_75, osnet_x1_0

# Embedder 协议符合性静态钉死：OsnetEmbedder 可赋给 type[Embedder]
# （结构化子类型——实例满足 reid_assoc.Embedder 协议；mypy 编译期校验）。
_EMBEDDER_CONFORMANCE: type[Embedder] = OsnetEmbedder


class _FakeModel(torch.nn.Module):
    """确定性投影 3*H*W -> 512（取展平前 512 元素，不足零填充）：输出可
    预测，验证预处理与 L2。（计划稿的 512-H*W 零填充在 256×128 输入下为
    负维度 torch.zeros——fake 自身 bug，实现无涉。）"""

    def forward(self, x):
        flat = x.flatten(1)[:, :512]
        out = torch.zeros(x.shape[0], 512)
        out[:, : flat.shape[1]] = flat
        return out


def test_preprocess_shape_and_normalization(monkeypatch):
    emb = OsnetEmbedder.__new__(OsnetEmbedder)  # 不加载真权重
    img = np.full((480, 200, 3), 127, dtype=np.uint8)
    batch = emb._preprocess([img, img])
    h, w = OSNET_INPUT_HW
    assert batch.shape == (2, 3, h, w)
    # 全灰图归一化后每通道 = (127/255 - mean) / std，数值确定
    c0 = (127 / 255 - 0.485) / 0.229
    assert batch[0, 0].mean() == pytest.approx(c0, abs=1e-4)


def test_preprocess_channel_order_bgr_to_rgb():
    """通道序契约锁定（审计修复 2026-09-13）：BGR=(255,0,0) 纯蓝输入 → 翻转后
    channel-0 = R 分支（值 0 → RGB 序 mean 0.485）。删掉 ``[..., ::-1]`` 翻转
    则 channel-0 命中 B=255 → (1−0.485)/0.229，本测试即红——全灰图对通道序
    色盲，锁不住这行与 Rust RGB 契约对齐的关键翻转。"""
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    img = np.zeros((480, 200, 3), dtype=np.uint8)
    img[..., 0] = 255  # BGR 布局的 B 通道
    batch = emb._preprocess([img])
    r = (0 / 255 - 0.485) / 0.229  # R=0：翻转后首通道
    b = (255 / 255 - 0.406) / 0.225  # B=255：翻转后末通道
    assert batch[0, 0].mean() == pytest.approx(r, abs=1e-4)
    assert batch[0, 2].mean() == pytest.approx(b, abs=1e-4)


def test_embed_l2_normalized_and_batched():
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.device = torch.device("cpu")
    emb.model = _FakeModel().eval()
    # 协议单 crop 接口
    single = emb.embed(np.full((60, 30, 3), 200, np.uint8))
    assert single.shape == (OSNET_DIM,)
    assert single.dtype == np.float32
    assert float(np.linalg.norm(single)) == pytest.approx(1.0, abs=1e-5)
    # iapx 原批量语义（embed_batch）
    out = emb.embed_batch([np.full((60, 30, 3), 200, np.uint8)] * 3)
    assert len(out) == 3 and out[0].shape == (OSNET_DIM,)
    for v in out:
        n = float(np.linalg.norm(v))
        assert n == pytest.approx(1.0, abs=1e-5)


class _OrderProbeModel(torch.nn.Module):
    """per-crop 可判别探针：输出 = (1, 预处理后 [0,0,0] 值, 0…)——常量填充 crop
    的 channel-0 值在归一化后随填充值单调，方向 (1, u) 唯一编码 crop 身份。"""

    def forward(self, x):
        out = torch.zeros(x.shape[0], 512)
        out[:, 0] = 1.0
        out[:, 1] = x[:, 0, 0, 0]
        return out


def test_embed_preserves_order_across_batch_boundary():
    """>64 crop 跨批次边界顺序保持（审计修复 2026-09-13）：``embed_batch`` 分块
    循环的顺序 append 若错位，embedding 与 detection 错配（strict zip 只查数量），
    全量语料 same/diff 分布整体错乱且无报错。70 crop（> ``_BATCH``=64，含
    第二块）逐位断言输出序 == 输入序。"""
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    emb.device = torch.device("cpu")
    emb.model = _OrderProbeModel().eval()
    vals = [1 + i * 3 for i in range(70)]  # 70 个互异常量填充值
    crops = [np.full((60, 30, 3), v, np.uint8) for v in vals]
    out = emb.embed_batch(crops)
    assert len(out) == 70
    for i, v in enumerate(vals):
        # 常量图 resize/翻转不改变值；channel-0 = R 分支归一化结果
        u = (v / 255 - 0.485) / 0.229
        assert out[i][1] / out[i][0] == pytest.approx(u, rel=1e-3), i


def test_embed_zero_area_returns_protocol_sentinel():
    """协议路径（``embed``）退化 crop → 全零 512-d 哨兵（协议契约：
    ``associate`` 见零范数即 id=0——与 iapx 批量接口 raise 的唯一分叉点）。"""
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    out = emb.embed(np.zeros((0, 10, 3), np.uint8))
    assert out.shape == (OSNET_DIM,)
    assert np.all(out == 0.0)


def test_embed_batch_rejects_zero_area():
    """批量路径（``embed_batch``）退化 crop ValueError fail-loud（iapx 原语义）。"""
    emb = OsnetEmbedder.__new__(OsnetEmbedder)
    with pytest.raises(ValueError, match="zero-area"):
        emb.embed_batch([np.zeros((0, 10, 3), np.uint8)])


# ---------------------------------------------------------------------------
# tag → (vendor 工厂, 权重路径) 映射单一数据源——模型指纹盘点
# ---------------------------------------------------------------------------

def test_weight_spec_mapping_covers_variants():
    # 枚举与消费方（iap Rust reid 轴 / py/tools gt-calibrate）对齐；各变体指向
    # 不同权重文件且都在盘上（权重盘为 jxl/sgcc 项目资产——jxl 惯例：缺盘
    # skip 而非 fail，同 reid.py 对 dinov2 权重的重依赖门）。
    # ft-v2 = jxl 域微调交付（2026-09-16，需求 C）：x0_75 架构 + 微调权重文件。
    # ft-v21 = 姿态漂移正对自挖 v2.1 追加交付（同日，见 jxl-notice 文档）。
    if not WEIGHTS_DIR.is_dir():
        pytest.skip(f"缺权重盘 {WEIGHTS_DIR}，跳过权重盘点")
    assert set(OSNET_WEIGHTS) == {
        "osnet-x0_75",
        "osnet-x1_0",
        "osnet-x0_75-ft-v2",
        "osnet-x0_75-ft-v21",
    }
    x075, x1, ft2 = (
        OSNET_WEIGHTS["osnet-x0_75"], OSNET_WEIGHTS["osnet-x1_0"],
        OSNET_WEIGHTS["osnet-x0_75-ft-v2"],
    )
    assert x075.build is osnet_x0_75 and x1.build is osnet_x1_0
    assert ft2.build is osnet_x0_75 and ft2.path.name == "osnet_x0_75_ft_v2.pth"
    assert "osnet_x0_75_msmt17_combineall" in x075.path.name
    assert "osnet_x1_0_msmt17_combineall" in x1.path.name
    assert len({x075.path, x1.path, ft2.path}) == 3
    assert all(p.is_file() for p in (x075.path, x1.path, ft2.path))


def test_default_tag_is_x1_0():
    # 质量优先裁决：默认 = 最强特征 x1_0（构造器缺省与映射一致）
    assert OSNET_DEFAULT_TAG == "osnet-x1_0"
    assert OSNET_DEFAULT_TAG in OSNET_WEIGHTS


def test_unknown_tag_rejected_before_load():
    # 未知 tag fail-loud（No Silent Degradation）；tag 解析先于权重加载，
    # device="cpu" 绕开 CUDA 断言 → 本测试零 GPU / 零权重 IO
    with pytest.raises(ValueError, match="bogus"):
        OsnetEmbedder("bogus", device="cpu")


def test_real_weight_load_and_embed_smoke():
    """真权重加载冒烟（jxl 重依赖门惯例：缺权重盘 / 缺 CUDA 即 skip）——
    覆盖 ``__new__`` 系测试触不到的加载路径：三态 unwrap + strict load +
    剥分类头 + GPU 推理 + L2 归一化。"""
    if not (WEIGHTS_DIR / "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth").is_file():
        pytest.skip("缺 osnet-x1_0 权重，跳过加载冒烟")
    if not torch.cuda.is_available():
        pytest.skip("缺 CUDA，跳过加载冒烟")
    emb = OsnetEmbedder(OSNET_DEFAULT_TAG)
    crop = np.random.RandomState(0).randint(0, 256, (256, 128, 3), dtype=np.uint8)
    out = emb.embed(crop)
    assert out.shape == (OSNET_DIM,)
    assert out.dtype == np.float32
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-3
