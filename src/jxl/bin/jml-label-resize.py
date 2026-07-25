#!/usr/bin/env python3

import argparse
from pathlib import Path

from jvi.geo.size2d import size_parse

from jxl.label.darknet.darknet_set import darknet_dump_labels
from jxl.label.hop import hop_load_labels
from jxl.label.meta import find_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="改变样本尺寸程序")
    parser.add_argument(
        "src_dir", metavar="SRC_DIR", type=Path, help="来源样本标注目录"
    )
    parser.add_argument(
        "dst_dir", metavar="DST_DIR", type=Path, help="目的样本标注目录"
    )
    parser.add_argument("meta_id", metavar="META_ID", type=int, help="元数据ID")
    parser.add_argument(
        "-s", "--size", type=str, default="640x640", help="目标样本图像尺寸"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    assert opt.src_dir.is_dir(), f"数据来源目录不存在: {opt.src_dir}"
    dst_size = size_parse(opt.size)
    assert dst_size, f"目标样本图像尺寸无效: {opt.size}"

    meta = find_meta(opt.meta_id, opt.src_dir).unwrap()

    print(f"加载目录: {opt.src_dir}")

    labels = hop_load_labels(opt.src_dir, opt.meta_id)

    assert len(labels) > 0

    dst_dir = opt.dst_dir

    # TODO: 坐标变换
    total = darknet_dump_labels(labels, dst_dir, meta)
    print(f"\n样本({total})生成完毕!")


if __name__ == "__main__":
    # catch_show_err(main, True)
    main()
