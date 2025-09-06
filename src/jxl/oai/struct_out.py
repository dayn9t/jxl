from jcx.text.txt_json import load_json
from openai import OpenAI, api_key
from pydantic import BaseModel

from jxl.common import JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


def main():
    # cfg =  load_json(JXL_OAI_DIR / "doubao_vl.json", LlmCfg).unwrap()
    cfg = load_json(JXL_OAI_DIR / "qwen_plus.json", LlmCfg).unwrap()

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    response = client.responses.parse(
        model=cfg.model,
        input=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        text_format=CalendarEvent,
    )

    event = response.output_parsed
    # print("model:", response.model)
    # print("total_tokens:", response.usage.total_tokens)
    print(response)
    print(event)


if __name__ == "__main__":
    main()
