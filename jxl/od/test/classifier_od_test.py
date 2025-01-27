from itertools import pairwise

from jcx.ui.key import Key
from jiv.geo.rectangle import Rect
from jiv.geo.size2d import Size
from jiv.image.io import load_images_in
from jiv.image.trace import trace_images, close_all_windows
from jml.od.classifier_od import *
from jml.iqa.diag_extractor import DiagExtractor, DIV_ROWS, DIV_COLS, DIV_SIZE


def main() -> None:
    model_dir = Path('/opt/ias/project/shtm/model/iqa')
    model_path = model_dir / 'moved.joblib'

    image_dir = Path('/home/jiang/ws/trash/outside/2022/n1/neg1/31011000602700301')
    # image_dir = Path('/home/jiang/ws/trash/outside/2022/n1/n/31010102100100201')
    # image_dir = Path('/home/jiang/ws/trash/outside/2022/n1/n/1')
    opt = ClassifierOpt((DIV_COLS * DIV_ROWS,), 3)
    model = ClassifierOd(model_path, opt)

    extractor = DiagExtractor(Rect.one(), DIV_COLS, DIV_ROWS, DIV_SIZE)

    images = load_images_in(image_dir)
    num_err = 0
    for i, (a, b) in enumerate(pairwise(images)):
        print(f'#{i} {images[i]}')
        v = extractor.extract(a, b).match_vec

        r = model(v)
        if r.top_index() == 1:
            print(f'  r: {r}')
            num_err += 1
            key, _ = trace_images([a, b], box_size=Size(1920, 540), auto_close=False)
            if key == Key.ESC:
                break

    total = len(images) - 1
    ratio = round(num_err / total, 4)
    print(f'错误率: {num_err}/{total}({ratio * 100}%)')
    close_all_windows()


if __name__ == '__main__':
    main()
