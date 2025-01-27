import torch
from safetensors.torch import load_file, save_file
from torch import Tensor


def export_safetensors() -> None:
    folder = "/home/jiang/ws/trash_arm/model/"
    # name = "yolov8n"
    name = "sort"
    name = "amount"
    src_path = folder + f"{name}.pth"
    dst_path = folder + f"{name}.safetensors"
    src_data = torch.load(src_path)

    # print(src_data['model'])
    save_file(src_data.state_dict(), dst_path)

    print("Done!")


def main() -> None:
    pass


if __name__ == "__main__":
    # main()
    export_safetensors()
