# 替换 & 关闭 torch.jit, 因为 JIT 与pyinstaller冲突


def script_method[T](fn: T, _rcb: object = None) -> T:
    return fn


def script[T](
    obj: T, optimize: bool = True, _frames_up: int = 0, _rcb: object = None
) -> T:
    return obj


import torch.jit  # noqa: E402  # monkeypatch 结构：no-op 替换须先定义
from loguru import logger  # noqa: E402


def disable_jit() -> None:
    # 运行时 monkeypatch：用 no-op 替换 torch.jit，规避 JIT 与 pyinstaller 冲突。
    torch.jit.script_method = script_method
    torch.jit.script = script  # type: ignore[assignment]
    logger.info("Disable torch JIT")
