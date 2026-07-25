from jcx.text.txt_json import load_json, to_json

from jxl.det.a2d import A2dResult
from tests.consts import META_DIR


def test_a2d_result():
    r = load_json(META_DIR / "a2d_info.json", A2dResult).unwrap()
    print(to_json(r))


if __name__ == "__main__":
    test_a2d_result()
