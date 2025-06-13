import onnx
from onnx.optimizer import optimize


def main() -> None:
    folder = "/opt/ias/project/shtm/model/cabin2/"
    src = "2024-01-30_cabin.onnx"
    dst = "2024-01-30_cabin_opt.onnx"
    # 加载ONNX模型
    model = onnx.load(folder + src)

    # 优化模型
    passes = ["fuse_bn_into_conv"]
    model = optimize(model, passes)

    # 保存优化后的模型
    onnx.save(model, folder + src)


if __name__ == "__main__":
    main()
