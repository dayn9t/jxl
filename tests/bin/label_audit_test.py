"""label_audit 单测: 聚类审计的纯逻辑部分."""

from jxl.bin.label_audit import BoxRow, cluster_cam


def _row(stem: str, cx: float, cy: float, w: float = 50.0, h: float = 120.0) -> BoxRow:
    return BoxRow(
        stem, stem.split("_")[2], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        (cx, cy, w, h),
    )


def test_cluster_cam_finds_static_cluster() -> None:
    """静态物(同位置跨日期高复现)应聚成一簇, 游走框应为噪声."""
    rows = (
        [_row(f"1_0_2026-06-{d:02d}_08-00-03.000_{i:05d}", 612, 364) for d in range(1, 6) for i in range(10)]
        + [_row(f"1_0_2026-06-01_09-00-03.000_{i:05d}", 80 + i * 90, 70 + i * 85) for i in range(12)]
    )
    clusters = cluster_cam(rows, eps=30.0, min_samples=20)
    assert len(clusters) == 1, f"静态物应聚成 1 簇, 游走框应为噪声, got {len(clusters)}"
    c = clusters[0]
    assert c["n_boxes"] == 50
    assert c["n_dates"] == 5
    assert abs(c["center"][0] - 612) < 1 and abs(c["center"][1] - 364) < 1


def test_cluster_cam_extent_covers_members() -> None:
    """extent 应覆盖全部成员框(供区域全扫清洗)."""
    rows = [_row(f"1_0_2026-06-{d:02d}_08-00-03.000_{i:05d}", 612 + i, 364 - i) for d in range(1, 5) for i in range(8)]
    (c,) = cluster_cam(rows, eps=30.0, min_samples=20)
    assert c["extent"][0] <= 612 - 25 and c["extent"][2] >= 612 + 7 + 25


def test_cluster_cam_noise_not_clustered() -> None:
    """分散框不足 min_samples 时无簇."""
    rows = [_row(f"1_0_2026-06-01_08-00-03.000_{i:05d}", i * 60, i * 55) for i in range(15)]
    assert cluster_cam(rows, eps=30.0, min_samples=20) == []
