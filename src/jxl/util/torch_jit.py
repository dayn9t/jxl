"""torch.jit 无操作 shim: 替换 & 关闭 torch.jit, 因为 JIT 与 pyinstaller 冲突.

提供与 torch.jit.script / torch.jit.script_method 同签名的无操作实现,
供 disable_jit() 运行时替换, 规避 JIT 编译与打包工具的冲突。
"""

import torch.jit
from loguru import logger


def script_method[T](fn: T, _rcb: object = None) -> T:
    """torch.jit.script_method 的无操作 shim (原样返回可调用对象)."""
    return fn


def script[T](obj: T, *, optimize: bool = True, _frames_up: int = 0, _rcb: object = None) -> T:  # noqa: ARG001, E501
    """torch.jit.script 的无操作 shim (原样返回对象).

    optimize/_frames_up/_rcb 仅用于匹配 torch.jit.script 签名, 不参与逻辑。
    """
    return obj


def disable_jit() -> None:
    """用无操作 shim 替换 torch.jit 的 JIT 入口 (pyinstaller 兼容性 monkeypatch).

    setattr 绕过静态类型检查 — 运行时替换 typed 模块的导出符号是刻意为之
    (直接赋值会触发 mypy method-assign, 故用 setattr)。
    """
    setattr(torch.jit, "script_method", script_method)  # noqa: B010
    setattr(torch.jit, "script", script)  # noqa: B010
    logger.info("Disable torch JIT")
