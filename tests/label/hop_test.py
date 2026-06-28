from jxl.label.hop import *
import pytest
from parse import parse  # type: ignore


# FIXME(pre-existing): 依赖外部数据目录 /home/jiang/1, 非自包含单元测试。待 PR-3 改造或删除。
@pytest.mark.skip(reason="pre-existing: requires external data dir /home/jiang/1")
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


# FIXME(pre-existing): 依赖外部数据目录 /var/ias/snapshot/..., 非自包含单元测试。
@pytest.mark.skip(reason="pre-existing: requires external data dir /var/ias/...")
def test_load_label_records():
    folder = "/var/ias/snapshot/shtm/n1/work"

    rs = hop_load_labels(folder, 31)
    print("rs:", len(rs))
