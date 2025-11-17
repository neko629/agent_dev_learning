import requests
import json

chat_url = "http://localhost:11434/api/chat"

chat_payload = {
    "model": "deepseek-r1:7b",
    "messages": [
        {
            "role": "user",
            "content": "简短的介绍一下什么是人工智能"
        }
    ],
    "tools": [],
    "stream": False,
}

response_chat = requests.post(chat_url, json=chat_payload)
if response_chat.status_code == 200:
    chat_response = response_chat.json()
    print("聊天响应:", json.dumps(chat_response, ensure_ascii=False, indent=2))
else:
    print("聊天请求失败:", response_chat.status_code, response_chat.text)