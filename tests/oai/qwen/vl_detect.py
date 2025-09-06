from jcx.text.txt_json import to_json, load_json
from openai import OpenAI
import base64
import time

from jxl.common import JXL_ASSERTS, JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg

# [物体定位](https://help.aliyun.com/zh/model-studio/vision/?spm=a2c4g.11186623.help-menu-2400256.d_0_2_0.501620b9pPPaJo#98e63a5a0akb9)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    cfg = load_json(JXL_OAI_DIR / "qwen/vl_7b.json", LlmCfg).unwrap()
    print("cfg:", to_json(cfg))

    file = JXL_ASSERTS / "person/p2.jpg"
    base64_image = encode_image(file)

    prompts = [
        "用框定位每个人位置，并描述其各自的特征，以JSON格式输出所有的bbox的坐标，",
        "用点定位每个人位置，并描述其各自的特征，以JSON格式输出所有的point的坐标，",
    ]
    prompt = prompts[0] + "不要输出```json```代码段。"

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
                    {"type": "text", "text": prompt},
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
