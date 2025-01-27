from jml.label.hop import *
from parse import parse  # type: ignore


def test_hop_label_path_of():
    a = Path('/tmp/a.txt')
    print(a.stem, type(a.stem))
    print(a.name, type(a.name))
    print(a.with_name('b.json'))

    print(hop_label_path_of(a, 11))
    r = parse('{}_s{}.{}', '11-11-11_s11.json')
    print(r)

    folder = Path('/home/jiang/1')
    for f in folder.rglob('*_s11.o'):
        print(f)


def load_label_records_test():
    folder = '/var/ias/snapshot/shtm/n1/work'

    rs = hop_load_labels(folder, 31)
    print('rs:', len(rs))
