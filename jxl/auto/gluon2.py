import os

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor  # type: ignore


def main() -> None:
    """允许输入同时包括图像和表格数据, 很强大, 但用处在哪里?"""

    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./tiny_motorbike_coco"
    # load_zip.unzip(zip_file, unzip_dir=download_dir)

    data_dir = os.path.join(download_dir, "tiny_motorbike")
    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")

    presets = "medium_quality"
    import uuid

    model_path = f"./tmp/{uuid.uuid4().hex}-quick_start_tutorial_temp_save"

    predictor = MultiModalPredictor(
        problem_type="object_detection",
        sample_data_path=train_path,
        presets=presets,
        path=model_path,
    )


if __name__ == "__main__":
    main()
