#!/home/jiang/py/jxl/.venv/bin/python
"""person Re-ID 同人聚类(HDBSCAN, 身份级).

对 OSNet Re-ID embedding 做 HDBSCAN(cosine) 聚类, 自动合并同一人的 crop,
输出 identity_map(每个 crop -> 身份 id, -1 为噪声/孤立). 供场景A整图身份指纹去重.
监控固定摄像头(几个人反复)场景下, Re-ID 是身份区分的唯一可靠手段(DINOv2 不行).

典型用法:
    person_dedup_reid /path/reid.npy /path/identity_out
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger
from sklearn.cluster import HDBSCAN

# typer CLI 惯用模式: 参数校验异常消息豁免噪声规则

app = typer.Typer(help="person Re-ID 同人聚类(HDBSCAN)")


@app.command()
def main(
    reid_npy: Annotated[Path, typer.Argument(help="reid_embeddings.npy")],
    out_dir: Annotated[Path, typer.Argument(help="输出目录(identity_map)")],
    min_cluster_size: Annotated[int, typer.Option(help="HDBSCAN min_cluster_size")] = 5,
) -> None:
    """HDBSCAN(cosine) 同人聚类, 输出 identity_map(供场景A身份指纹)."""
    emb = np.load(reid_npy).astype(np.float32)
    files = reid_npy.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    n = len(files)
    assert len(files) == n, f"emb {n} != files {len(files)}"
    logger.info("加载 {} reid embedding, dim={}", n, emb.shape[1])

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine")
    labels = hdb.fit_predict(emb)
    n_clusters = len({int(x) for x in labels if x >= 0})
    n_noise = int((labels < 0).sum())
    logger.info("HDBSCAN(min_size={}): 身份簇={}, 噪声(孤立)={}", min_cluster_size, n_clusters, n_noise)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "identity_map.npy", labels.astype(np.int64))
    (out_dir / "identity_files.txt").write_text("\n".join(files), encoding="utf-8")
    logger.info("完成: identity_map({}) -> {}", n, out_dir)


if __name__ == "__main__":
    app()
