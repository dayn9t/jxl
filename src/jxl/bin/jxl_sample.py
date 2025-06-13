#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jxl.label.darknet.darknet_set import darknet_dump_labels
from jxl.label.hop import hop_load_labels
from jxl.label.io import dump_label_prop
from jxl.label.meta import find_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="样本生成程序")
    parser.add_argument("src_dir", metavar="SRC_DIR", type=Path, help="来源标注目录")
    parser.add_argument("dst_dir", metavar="DST_DIR", type=Path, help="目的样本目录")
    parser.add_argument("meta_id", metavar="META_ID", type=int, help="元数据ID")
    parser.add_argument("-c", "--category", type=str, help="指定类别")
    parser.add_argument("-p", "--prop", type=str, help="属性值")
    parser.add_argument("-P", "--prefix", type=str, default="", help="样本文件前缀")
    parser.add_argument(
        "-k", "--keep_dst_dir", action="store_true", help="保留目标目录, 不重建"
    )
    parser.add_argument(
        "-C", "--crop_roi", action="store_true", help="裁剪ROI, 只对检测任务有效"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    opt = parser.parse_args()

    assert opt.src_dir.is_dir(), f"数据来源目录不存在: {opt.src_dir}"

    meta = find_meta(opt.meta_id, opt.src_dir).unwrap()

    print(f"加载目录: {opt.src_dir}")

    labels = hop_load_labels(opt.src_dir, opt.meta_id)

    assert len(labels) > 0

    dst_dir = opt.dst_dir
    if opt.prop:
        cat_id = meta.cat_meta(name=opt.category).id
        total = dump_label_prop(
            labels, dst_dir, cat_id, opt.prop, opt.keep_dst_dir, opt.prefix
        )
    else:
        total = darknet_dump_labels(labels, dst_dir, opt.crop_roi, opt.keep_dst_dir)
    print(f"\n样本({total})生成完毕!")


if __name__ == "__main__":
    # catch_show_err(main, True)
    main()
