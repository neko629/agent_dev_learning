# 流式输出

import requests # type: ignore
import json

# 设置 API 端点
generate_url = "http://localhost:11434/api/generate"

# 示例数据
generate_payload = {
    "model": "deepseek-r1:7b",
    "prompt": "请生成一个关于人工智能的简短介绍。",
    "keep_alive": "30s",
    "options": {
        "temperature": 0.6,
    }
}

# 调用生成接口
response_generate = requests.post(generate_url, json=generate_payload, stream=True)  # 在这里添加stream=True
if response_generate.status_code == 200:
    # 处理流式响应
    for line in response_generate.iter_lines():
        if line:
            try:
                # 解码并解析每一行的 JSON
                response_json = json.loads(line.decode('utf-8'))
                if 'response' in response_json:
                    print(response_json['response'], end='', flush=True)

                # 检查 response_json 字典中是否存在键 'done'，并且其值是否为 True。如果这个条件成立，表示生成的响应已经完成。
                if response_json.get('done', False):
                    print('\n\n完整响应:', json.dumps(response_json, ensure_ascii=False, indent=2))
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
else:
    print("生成请求失败:", response_generate.status_code, response_generate.text)