from pathlib import Path
from typing import Final

JXL_ASSERTS: Final = Path(__file__).parent.parent.parent / "assets"

JXL_META_DIR: Final = JXL_ASSERTS / "meta"
JXL_OAI_DIR: Final = JXL_ASSERTS / "oai"


def test_dis():
    assert JXL_ASSERTS.is_dir()
    assert JXL_META_DIR.is_dir()
    assert JXL_OAI_DIR.is_dir()
