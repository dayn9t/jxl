# 替换 & 关闭 torch.jit, 因为 JIT 与pyinstaller冲突
def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit


def disable_jit() -> None:
    torch.jit.script_method = script_method
    torch.jit.script = script
    print('Disable torch JIT')
