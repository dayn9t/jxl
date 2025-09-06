from jcx.time.stop_watch import StopWatch
from openai import OpenAI
import os
from pydantic import BaseModel
from jcx.text.txt_json import to_json


class Step(BaseModel):
    explanation: str
    output: str


class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str


def main():
    client = OpenAI(
        api_key="a0f8cf60-dede-44ca-bf0a-581854496fa4",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    watch = StopWatch()
    watch.start()
    completion = client.beta.chat.completions.parse(
        model="doubao-seed-1-6-250615",
        messages=[
            {"role": "system", "content": "你是一位数学辅导老师。"},
            {"role": "user", "content": "使用中文解题: 8x + 9 = 32 and x + y = 1"},
        ],
        response_format=MathResponse,
        extra_body={
            "thinking": {
                "type": "disabled"  # 不使用深度思考能力
                #"type": "enabled" # 使用深度思考能力
            }
        }
    )
    print("elapsed:", watch.stop());
    resp = completion.choices[0].message.parsed
    # 打印 JSON 格式结果
    # print(resp.model_dump_json(indent=2))
    print(to_json(completion))


if __name__ == "__main__":
    main()
