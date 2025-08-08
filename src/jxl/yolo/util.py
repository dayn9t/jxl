from ultralytics.utils import SETTINGS


def yolo_set_weights_dir(path: str) -> None:
    SETTINGS["weights_dir"] = path
