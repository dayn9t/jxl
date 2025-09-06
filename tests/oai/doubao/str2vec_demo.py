import os
from volcenginesdkarkruntime import Ark

def main():
    # 初始化客户端
    client = Ark(
        # 从环境变量中读取您的方舟API Key
        api_key="a0f8cf60-dede-44ca-bf0a-581854496fa4",
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    response = client.embeddings.create(
        model="doubao-embedding-text-240715",
        input="Function Calling 是一种将大模型与外部工具和 API 相连的关键功能",
        encoding_format="float"
    )
    # 打印结果
    print(f"向量维度: {len(response.data[0].embedding)}")
    print(f"前10维向量: {response.data[0].embedding[:10]}")

if __name__ == "__main__":
    main()