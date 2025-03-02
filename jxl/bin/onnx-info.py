from pathlib import Path

import onnxruntime as ort
import onnx

def main1(model_path: Path):

    providers = ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        session_options=session_options,
        providers=providers
    )

    inputs_info = session.get_inputs()
    print("模型的输入信息:")
    for input in inputs_info:
        print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")

    outputs_info = session.get_outputs()
    print("\n模型的输出信息:")
    for output in outputs_info:
        print(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")


def main2(model_path: Path):

    model = onnx.load(model_path)
    for imp in model.opset_import:
        print(f"打印OpSet导入信息: {imp.domain}, Version: {imp.version}")
    # 验证模型是否有效
    onnx.checker.check_model(model)
    print("模型的输入信息:")
    for input in model.graph.input:
        print(f"Name: {input.name}, Type: {input.type}")
    print("\n模型的输出信息:")
    for output in model.graph.output:
        print(f"Name: {output.name}, Type: {output.type}")


if __name__ == "__main__":

    folder = Path("/opt/howell/iws/current/ias/model/cabin2")
    model_path = folder / "2024-03-16_cabin.onnx"

    main1(model_path)
