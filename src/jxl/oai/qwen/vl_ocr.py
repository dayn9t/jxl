from jcx.text.txt_json import to_json, load_json
from openai import OpenAI
import base64
import time

from jxl.common import JXL_ASSERTS, JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    cfg = load_json(JXL_OAI_DIR / "qwen_ocr_plus.json", LlmCfg).unwrap()
    print("cfg:", to_json(cfg))

    file = JXL_ASSERTS / "s4/signs/电信1.jpg"
    base64_image = encode_image(file)

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.api_base,
    )

    # 记录请求开始时间
    start_time = time.time()

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            #{
            #    "role": "system",
            #    "content": [{"type": "text", "text": "You are a helpful assistant."}],
            #},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        "min_pixels": 28 * 28 * 4,
                        # 输入图像的最大像素阈值，超过该值图像会按原比例缩小，直到总像素低于max_pixels
                        "max_pixels": 28 * 28 * 8192
                    },
                    #{"type": "text", "text": "?"},
                ],
            },
        ],
    )

    # 记录请求结束时间并计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("model:", completion.model)
    print("total_tokens:", completion.usage.total_tokens)
    print("请求时间: {:.2f} 秒".format(elapsed_time))
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
