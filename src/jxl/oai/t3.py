from jcx.text.txt_json import load_json, to_json
from loguru import logger
from openai import OpenAI

from jxl.common import JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg


def main() -> None:
    cfg = load_json(JXL_OAI_DIR / "qwen_flash.json", LlmCfg).unwrap()

    logger.info("cfg: {}", to_json(cfg))

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
    logger.info("model: {}", completion.model)
    usage = completion.usage
    logger.info("total_tokens: {}", usage.total_tokens if usage else 0)
    logger.info("{}", json_string)


if __name__ == "__main__":
    main()
