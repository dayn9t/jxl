import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from jcx.text.txt_json import load_json, save_json

from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path


class FileInfo(BaseModel):
    image: Path


class SampleTab(BaseModel):
    files: List[FileInfo] = []
    dist_mat: Dict[int, Dict[int, float]] = {}
    likelihood_mat: Dict[int, Dict[int, float]] = {}

    def cale_likelihood_mat(self, model):
        """计算似然矩阵"""

        for i, m in self.dist_mat.items():
            for j, dist in m.items():
                im1 = self.files[i].image
                im2 = self.files[j].image
                s = cale_similarity(model, im1, im2)
                if i not in self.likelihood_mat:
                    self.likelihood_mat[i] = {}
                self.likelihood_mat[i][j] = s
            print("#%d" % i, len(self.likelihood_mat[i]))


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
    # 示例图像路径
    image_path1 = "/home/jiang/py/jxl/assets/images/10-20-49.241_1.jpg"
    image_path2 = "/home/jiang/py/jxl/assets/images/10-20-50.041_1.jpg"
    image_path2 = "/home/jiang/py/jxl/assets/images/10-21-23.839_1.jpg"

    # 加载预训练的 ResNet 模型
    model = models.resnet18(pretrained=True)
    model = model.to("cuda")
    # 去掉最后一层分类器，只保留特征提取部分
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # 计算相似度
    # similarity = cale_similarity(model, image_path1, image_path2)
    # print(f"图像相似度: {similarity}")

    cale(model)


def cale(model):
    src_file = "/mnt/data/var/howell/s4/ias/meta_shop/d1/n1/2_images.json"
    dst_file = "/mnt/data/var/howell/s4/ias/meta_shop/d1/n1/2_images_like.json"
    print("load json file:", src_file)
    sample_tab = load_json(src_file, SampleTab).unwrap()
    sample_tab.cale_likelihood_mat(model)
    save_json(
        sample_tab,
        dst_file,
    )
    print("Done:", dst_file)


if __name__ == "__main__":
    main()
