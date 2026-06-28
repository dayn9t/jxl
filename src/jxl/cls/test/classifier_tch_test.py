from pathlib import Path

from jcx.sys.fs import files_in
from jvi.image.image_nda import ImageNda
from jvi.image.trace import trace_image

from jxl.cls.classifier import ClassifierOpt
from jxl.cls.classifier_tch import ClassifierTch


def show_main() -> None:
    model_path = Path("/opt/ias/project/shtm/model/cabin/can-amount")
    folder = Path("/home/jiang/ws/trash/can-amount/dates/2023-03-17/5")
    opt = ClassifierOpt((224, 224), 6)

    classifier = ClassifierTch(model_path, opt)

    for file in files_in(folder, ".jpg"):
        im: ImageNda = ImageNda.load(file)
        r = classifier(im)

        print("top_class :", r.top())  # noqa: T201
        print("confidences:", r.confidences())  # noqa: T201

        trace_image(im)


if __name__ == "__main__":
    show_main()
