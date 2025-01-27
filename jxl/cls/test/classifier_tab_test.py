from jcx.ui.key import Key
from jiv.image.io import load_images_in
from jiv.image.trace import trace_image
from jxl.cls.classifier_tab import *


def main(test_model: int = 1) -> None:
    from jxl.iqa.diag_extractor import sharpness, chroma

    model_dir = Path('/opt/ias/project/shtm/model/iqa')

    if test_model == 1:
        model_path = model_dir / 'clearness'
        image_dir = Path('/home/jiang/ws/diagnosis/clearness/samples/2')
        fun = sharpness
        opt = ClassifierOpt((256,), 3)
    else:
        model_path = model_dir / 'chroma'
        image_dir = Path('/home/jiang/ws/diagnosis/chroma/samples/1')
        fun = chroma
        opt = ClassifierOpt((256,), 2)

    model = ClassifierTab(model_path, opt)

    images = load_images_in(image_dir)
    for im in images:
        v = fun(im)
        r = model(v)
        print(type(r), r.top())
        key, _ = trace_image(im)
        if key == Key.ESC:
            break


if __name__ == '__main__':
    main()
