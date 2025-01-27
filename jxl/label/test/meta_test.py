from jcx.text.txt_json import try_load_json
from jxl.label.meta import *


def test_meta_format():
    """所有meta文件都应该在这里检验"""

    folder = Path("/project")
    files = [
        # 'sgcc/meta/smoke_fire_meta.json',
        # 'cnooc/meta/scene_meta.json',
        # 'shdt/meta/lift_meta.json',
        "shtm/meta/cabin_meta.json",
        "shtm/meta/diagnosis_meta.json",
        # 'shtm/meta/aipo/cabin-meta.json',
        # 'shtm/meta/aipo/camera-meta.json',
        "shjs/meta/jewelry_meta.json",
    ]
    for file in files:
        file = folder / file
        print("meta file:", file)
        assert file.is_file(), f"{file} not found"
        meta = try_load_json(file, LabelMeta)
        assert meta.is_some()
