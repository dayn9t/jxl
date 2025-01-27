#!/opt/ias/env/bin/python

import argparse
from pathlib import Path

from jcx.sys.fs import move_file
from jcx.ui.key import Key
from jiv.geo.size2d import Size
from jiv.gui.record_viewer import RecordViewer, load_dir_records


class Sorter(RecordViewer):

    def __init__(self, title: str, dst_dir: Path):
        super().__init__(title, Size(244 * 3 * 2, 244 * 2))
        self.dst_dir = dst_dir
        self.key_class = {
            ord("e"): '0',
            ord("q"): '1',
            Key.BLANK: 'pending'
        }

        self.help_msgs.extend([
            "分类按键表：",
            "  [e] 三图片一致(变化部分低于1/4)",
            "  [q] 最新图片变化(变化部分大于1/2)",
            "  [Space] 待定",
        ])

    def on_key(self, key: int):
        """按键处理程序"""
        cat = self.key_class.get(key, '')
        if cat:
            self._move_class(cat)
            self.jump_to(1)
            return 0
        return super().on_key(key)

    def _move_class(self, cat: str):
        """图片移动到指定类别"""
        r = self.record()
        dst = Path(self.dst_dir, cat, r.path.name)
        print('  %s => %s' % (r.path, dst))
        move_file(r.path, dst).unwrap()
        r.path = dst


# args = /home/jiang/ws/scene/cnooc/sources/31010102100700101 -v

def main():
    parser = argparse.ArgumentParser(description='图片人工分类')
    parser.add_argument('folder', type=Path, help='来源图片目录')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    opt = parser.parse_args()

    rs = load_dir_records(opt.folder)
    if len(rs) < 1:
        print('No files found')
        return

    win = Sorter('files://' + str(opt.folder), Path(opt.folder, 'classes'))
    win.set_records(rs)

    print('image number:', win.image_count())
    win.run()
    print('Done')


if __name__ == '__main__':
    main()
