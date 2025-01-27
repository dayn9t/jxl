import cv2
from jcx.sys.fs import file_names_in
from jcx.ui.key import Key
from jvi.geo.size2d import SIZE_FHD
from jvi.image.proc import to_color
from jvi.image.trace import trace_images, close_all_windows
from jxl.seg.cd.open_cd import *


def demo_change_detector() -> None:
    root = Path('/home/jiang/ws/cnooc/cd/dataset/test')
    model_file = Path('/home/jiang/ws/cnooc/cd/model_dir/landscape_cd.pth')

    opt = SegOpt((1024, 1024), 0.5, 0.7)
    model = OpenCD(model_file, opt, 'cuda')

    dirs = [root / 'A', root / 'B']

    names = file_names_in(dirs[0], '.png')
    for i, name in enumerate(names):
        images = [cv2.imread(str(d / name)) for d in dirs]

        result = model.forward_np(images)
        print(f'#{i} {name} range[{np.amin(result)},{np.amax(result)}], {result.dtype}')

        mask = ImageNda(data=result)
        mask = to_color(mask)
        images1 = [ImageNda(data=im) for im in images] + [mask]
        key, _ = trace_images(images1, 'A-B-Mask', box_size=SIZE_FHD, auto_close=False)
        if key == Key.ESC:
            break
    print('DONE!')
    close_all_windows()


if __name__ == '__main__':
    demo_change_detector()
