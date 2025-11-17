from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama'
)

messages = [
    {
        'role': 'user',
        'content': '简短的介绍一下什么是人工智能, 大约200字左右。'
    }
]

try:
    stream = client.chat.completions.create(
        model='deepseek-r1:7b',
        messages=messages,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='', flush=True)

except Exception as e:
    print(f"请求失败: {e}")