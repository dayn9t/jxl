from pathlib import Path
from typing import Optional

from jcx.sys.fs import StrPath
from jml.label.io import label_path_of

"""IAS文件系统相关内容, 不设计具体格式"""


def ias_label_path_of(img_file: StrPath, meta_id: int) -> Path:
    """获取图像对应的IAS报警消息文件路径"""
    return label_path_of(img_file, 'ias', meta_id, '.json')  # TODO: IAS常量


def ias_image_path_of_label(label_file: StrPath, ext: str) -> Optional[Path]:
    """标签文件对应的图像路径"""
    p = str(label_file)
    i = p.rfind('_')
    if i < 0:
        return None
    p = p[:i] + ext
    return Path(p)
