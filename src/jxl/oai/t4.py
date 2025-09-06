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
    cfg = load_json(JXL_OAI_DIR / "qwen_vl_plus.json", LlmCfg).unwrap()
    print("cfg:", to_json(cfg))

    file = JXL_ASSERTS / "person/p2.jpg"
    base64_image = encode_image(file)

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    # 记录请求开始时间
    start_time = time.time()

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "图中描绘的是什么景象?"},
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
