from pathlib import Path

import cv2
from loguru import logger


def calculate_image_sharpness(image_path: str) -> float:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cv2.imread 无法读取图像: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def main() -> None:
    folder = "/home/jiang/ws/trash/dates/2023-04-10/image"
    file = "n1_31010510200500202_2023-04-10_10-05-07.195.jpg"
    path = Path(folder, file)

    m = calculate_image_sharpness(str(path))
    logger.info(f"{m}")


if __name__ == "__main__":
    main()
