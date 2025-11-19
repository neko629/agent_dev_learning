from openai import OpenAI
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key = 'ollama'
)

chat_completion = client.chat.completions.create(
    model="deepseek-r1:7b",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function that checks if a number is prime."
        }
    ]
)

print(chat_completion)