import torch
from safetensors.torch import load_file, save_file
from torch import Tensor


def rename(name: str) -> str:
    name = name.replace("model.0.", "net.b1.0.")
    name = name.replace("model.1.", "net.b1.1.")
    name = name.replace("model.2.m.", "net.b2.0.bottleneck.")
    name = name.replace("model.2.", "net.b2.0.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.4.m.", "net.b2.2.bottleneck.")
    name = name.replace("model.4.", "net.b2.2.")
    name = name.replace("model.5.", "net.b3.0.")
    name = name.replace("model.6.m.", "net.b3.1.bottleneck.")
    name = name.replace("model.6.", "net.b3.1.")
    name = name.replace("model.7.", "net.b4.0.")
    name = name.replace("model.8.m.", "net.b4.1.bottleneck.")
    name = name.replace("model.8.", "net.b4.1.")
    name = name.replace("model.9.", "net.b5.0.")
    name = name.replace("model.12.m.", "fpn.n1.bottleneck.")
    name = name.replace("model.12.", "fpn.n1.")
    name = name.replace("model.15.m.", "fpn.n2.bottleneck.")
    name = name.replace("model.15.", "fpn.n2.")
    name = name.replace("model.16.", "fpn.n3.")
    name = name.replace("model.18.m.", "fpn.n4.bottleneck.")
    name = name.replace("model.18.", "fpn.n4.")
    name = name.replace("model.19.", "fpn.n5.")
    name = name.replace("model.21.m.", "fpn.n6.bottleneck.")
    name = name.replace("model.21.", "fpn.n6.")
    name = name.replace("model.22.", "head.")
    return name


def show(tensors: dict, title: str) -> None:
    print(title, type(tensors), len(tensors.items()))
    for k, v in tensors.items():
        # print('\t', str(k), v.shape)
        pass

    # return list(tensors.keys()).sort()


def export_safetensors() -> None:
    folder = "/home/jiang/ws/trash_arm/model/"
    # name = "yolov8n"
    name = "cabin5"
    src_path = folder + f"{name}.pt"
    dst_path = folder + f"{name}.safetensors"

    # src_path = "/home/jiang/ws/trash/cabin/model_dir/cabin.pt"
    src_data = torch.load(src_path)

    # print(src_data['model'])
    src_data = src_data["model"].state_dict().items()
    src_data = dict(src_data)
    show(src_data, "src:")
    dst_data = {rename(k): t for k, t in src_data.items()}
    # print(src_data["model"])
    save_file(dst_data, dst_path)
    show(dst_data, "dst:")
    print("Done!")


"""
def main() -> None:
    folder = "/home/jiang/ml/model/candle-yolo-v8/"
    # name = "yolov8n"
    name = "cabin5"
    src_path = folder + f"{name}.pt"
    dst_path = folder + f"dst/{name}.safetensors"
    ref_path = folder + f"{name}.safetensors"

    # src_path = "/home/jiang/ws/trash/cabin/model_dir/cabin.pt"
    src_data = torch.load(src_path)
    ref_data = load_file(ref_path)
    show(ref_data, "ref:")

    # print(src_data['model'])
    src_data = src_data['model'].state_dict().items()
    src_data = dict(src_data)
    show(src_data, "src:")
    dst_data = {rename(k): t for k, t in src_data.items()}
    # print(src_data["model"])
    save_file(dst_data, dst_path)
    show(dst_data, "dst:")

    assert len(ref_data) == len(dst_data)
    print('ref/dst len:', len(dst_data))
    i = 0
    for k, v in ref_data.items():
        assert k in dst_data
        if not torch.equal(v, dst_data[k]):
            # if not k.endswith('num_batches_tracked'):
            print(f'#{i}', k, v, dst_data[k])
            i += 1
            a = Tensor([0.])
            b = torch.ones((), dtype=torch.int64)
            assert a == v
            assert b == dst_data[k]
    print('Done!')
"""

if __name__ == "__main__":
    # main()
    export_safetensors()
