from jcx.text.txt_json import load_json, to_json
from openai import OpenAI

from jxl.common import JXL_OAI_DIR
from jxl.oai.types1 import LlmCfg


def main():
    cfg = load_json(JXL_OAI_DIR / "qwen_plus.json", LlmCfg).unwrap()

    print("cfg:", to_json(cfg))

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model=cfg.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    print("model:", to_json(completion.model))
    print("completion:", to_json(completion))


if __name__ == "__main__":
    main()
