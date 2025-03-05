import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from jcx.text.txt_json import load_json, save_json
from jcx.sys.fs import files_in
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import numpy as np
from jxl.common import JXL_ASSERTS, JXL_IMAGES_DIR


class FileInfo(BaseModel):
    image: Path


class SampleTab(BaseModel):
    files: List[FileInfo] = []
    dist_mat: Dict[int, Dict[int, float]] = {}

    def cale_likelihood_mat(self, model, dst_dir: Path):
        """计算似然矩阵"""
        items = sorted(self.dist_mat.items())

        for i, m in items:
            dst_file = dst_dir / f"{i}.json"
            if dst_file.is_file():
                print("skip:", dst_file)
            else:
                likelihood_map: Dict[int, float] = {}
                for j, dist in m.items():
                    im1 = self.files[i].image
                    im2 = self.files[j].image
                    s = cale_similarity(model, im1, im2)
                    likelihood_map[j] = float(s)
                print("#%d" % i, len(likelihood_map))

                save_json(likelihood_map, dst_file)


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# 加载图像
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to("cuda")
    return image


# 提取图像特征
def extract_features(model, image):
    with torch.no_grad():
        features = model(image)
        # print("features:", features)
        features = features.squeeze().cpu().numpy()
    return features


# 计算余弦相似度
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cale_similarity(model, image_path1, image_path2):
    # 加载并提取特征
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    features1 = extract_features(model, image1)
    features2 = extract_features(model, image2)

    # 计算相似度
    similarity = cosine_similarity(features1, features2)
    return similarity


def main():

    files = files_in(JXL_IMAGES_DIR, ".jpg")

    print("files:", JXL_IMAGES_DIR)

    # 加载预训练的 ResNet 模型
    model = models.resnet18(pretrained=True)
    model = model.to("cuda")
    # 去掉最后一层分类器，只保留特征提取部分
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    num_files = len(files)
    similarity_matrix = np.zeros((num_files, num_files), dtype=int)

    for i in range(num_files):
        for j in range(i + 1, num_files):
            similarity = cale_similarity(model, files[i], files[j])
            similarity = int(similarity * 100)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix

    print("Similarity Matrix:")
    print(similarity_matrix)
    # cale(model)


def cale(model):
    src_file = "/var/howell/s4/ias/meta_shop/d1/n1/2_images.json"
    dst_dir = Path("/var/howell/s4/ias/meta_shop/d1/n1/similarity")

    print("load json file:", src_file)
    sample_tab = load_json(src_file, SampleTab).unwrap()
    sample_tab.cale_likelihood_mat(model, dst_dir)

    print("Done!")


if __name__ == "__main__":
    main()
