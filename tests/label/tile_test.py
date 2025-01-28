from jvi.gui.record_viewer import RecordViewer
from jxl.label.tile import *


def load_tiles_test():
    folder = "/home/jiang/ws/trash/dates/2023-02-15"
    meta_id = 31
    category = 1
    prop = "amount"
    rs = load_tiles(folder, meta_id, category, prop)

    size = Size(1536, 864)
    obj_size = Size(256, 288)
    w = size.width // obj_size.width
    h = size.height // obj_size.height
    n = w * h

    rects = [r.erode(8) for r in Rect.from_size(size).to_tiles(w, h, need_round=True)]

    rs = [TileRecord.new(size, rects, rs[i : i + n]) for i in range(0, len(rs), n)]

    win = RecordViewer("tiles")
    win.set_records(rs)

    print("image number:", win.image_count())
    win.run()


if __name__ == "__main__":
    load_tiles_test()
