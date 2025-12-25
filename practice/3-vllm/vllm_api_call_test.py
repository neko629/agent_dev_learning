# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --host 0.0.0.0 \
#     --port 8030 \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 2048


from openai import OpenAI

# 1. 初始化客户端
client = OpenAI(
    # 将 IP 换成你服务器的实际 IP 地址
    # 注意：必须加上 "/v1" 后缀
    base_url="http://127.0.0.1:8030/v1",

    # vLLM 默认不验证 key，但在客户端库中必须填一个非空字符串，通常用 "EMPTY"
    api_key="EMPTY",
)

# 2. 获取当前服务端的模型列表（可选，用于验证连接）
models = client.models.list()
print(f"当前可用模型: {models.data[0].id}")
model_name = models.data[0].id

# 3. 发起对话请求 (Chat Completion)
try:
    response = client.chat.completions.create(
        model=model_name,  # 必须与服务器启动的模型名称一致
        messages=[
            {"role": "system", "content": "你是一个乐于助人的AI助手。"},
            {"role": "user", "content": "简单介绍一下你自己。"},
        ],
        temperature=0.7,
        max_tokens=512,
        stream=False  # 如果设为 True，需要迭代处理返回
    )

    print("回答内容：")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"调用失败: {e}")