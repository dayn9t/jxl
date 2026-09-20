"""jxl_viewer 导入链注册回归测试。

迁移后 LabelSet 注册改为 import 副作用：HopSet 由 jxl.label.hop 注册、
DarknetSet 由 vlabel.formats 注册，而 open_label_set 工厂只认已注册实现。
jxl_viewer 必须显式携带这两个注册 import，否则 `-f hop` / `-f darknet`
及 hop_m{id} 自动格式探测全部失效（返回 Err "not found" 后 unwrap panic）。

用子进程隔离：本进程内其他测试可能已 import 注册模块，无法复现回归。
"""

import subprocess
import sys

_PROBE = """
from pathlib import Path

import jxl.bin.jxl_viewer  # noqa: F401
from vlabel.dataset import LabelFormat, open_label_set

for fmt in (LabelFormat.HOP, LabelFormat.DARKNET):
    result = open_label_set(Path("/nonexistent"), fmt, 0)
    assert "not found" not in str(result), f"{fmt} not registered: {result}"
"""


def test_viewer_import_chain_registers_label_sets() -> None:
    """导入 jxl_viewer 后 HOP/DARKNET 必须已注册进 open_label_set 工厂."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
