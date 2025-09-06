from jcx.text.txt_json import load_json, to_json
from openai import OpenAI
import os

from jxl.common import JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg


def main():
    cfg = load_json(JXL_OAI_DIR / "qwen_flash.json", LlmCfg).unwrap()

    print("cfg:", to_json(cfg))

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": "请抽取用户的姓名与年龄信息，以JSON格式返回"},
            {
                "role": "user",
                "content": "大家好，我叫刘五，今年34岁，邮箱是liuwu@example.com，平时喜欢打篮球和旅游",
            },
        ],
        response_format={"type": "json_object"},
    )

    json_string = completion.choices[0].message.content
    print("model:", completion.model)
    print("total_tokens:", completion.usage.total_tokens)
    print(json_string)


if __name__ == "__main__":
    main()
