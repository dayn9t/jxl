import base64
import time

from jcx.text.txt_json import load_json, to_json
from openai import OpenAI

from jxl.common import JXL_ASSERTS, JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    cfg = load_json(JXL_OAI_DIR / "qwen/ocr_plus.json", LlmCfg).unwrap()
    print("cfg:", to_json(cfg))

    file = JXL_ASSERTS / "s4/signs/电信1.jpg"
    base64_image = encode_image(file)

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    # 记录请求开始时间
    start_time = time.time()

    # openai 存根不建模 Qwen VL 的 min_pixels/max_pixels 厂商扩展（content item 级），
    # 属第三方 stub 限制，按逐条理由豁免。
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    "min_pixels": 28 * 28 * 4,
                    # 输入图像的最大像素阈值，超过该值图像会按原比例缩小，直到总像素低于max_pixels
                    "max_pixels": 28 * 28 * 8192,
                },
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,  # type: ignore[arg-type]  # Qwen VL 厂商扩展，见上
    )

    # 记录请求结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("model:", completion.model)
    assert completion.usage is not None
    print("total_tokens:", completion.usage.total_tokens)
    print(f"请求时间: {elapsed_time:.2f} 秒")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
