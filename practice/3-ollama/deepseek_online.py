
from openai import OpenAI

dp_api_key = ''

client = OpenAI(
    base_url='https://api.deepseek.com',
    api_key=dp_api_key
)

response = client.chat.completions.create(
   # model="deepseek-chat",
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "你是一个每句话都要带上歌词的聊天机器人。还会解释是什么歌里的"},
        {"role": "user", "content": "Hello"},
    ],
    temperature=1.0,
    stream=False
)

print(response.choices[0].message.content)