
from jxl.label.meta import *
from tests.consts import META_DIR
from jcx.sys.fs import files_in

def test_meta_format():
    """所有meta文件都应该在这里检验"""

    folder = META_DIR
    files = files_in(folder, ".json")
    for file in files:
        file = folder / file
        print("\nmeta file:", file)
        assert file.is_file(), f"{file} not found"
        meta = load_json(file, LabelMeta).ok()
        assert meta.is_some()
