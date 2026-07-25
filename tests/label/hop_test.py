from pathlib import Path

from parse import parse

from jxl.label.hop import hop_label_path_of, hop_load_labels


def test_hop_label_path_of():
    a = Path("/tmp/a.txt")
    print(a.stem, type(a.stem))
    print(a.name, type(a.name))
    print(a.with_name("b.json"))

    print(hop_label_path_of(a, 11))
    r = parse("{}_s{}.{}", "11-11-11_s11.json")
    print(r)

    folder = Path("/home/jiang/1")
    for f in folder.rglob("*_s11.o"):
        print(f)


def test_load_label_records():
    folder = Path("/var/ias/snapshot/shtm/n1/work")
    # 依赖本地标注数据集, 数据缺失时跳过.
    if not (folder / "hop_m31").is_dir():
        import pytest

        pytest.skip(f"local label dataset not available: {folder / 'hop_m31'}")

    rs = hop_load_labels(folder, 31)
    print("rs:", len(rs))
