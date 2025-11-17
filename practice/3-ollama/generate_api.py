import requests # type: ignore
import json

# 设置 API 端点
generate_url = "http://localhost:11434/api/generate"    # 这里需要根据实际情况进行修改

# 示例数据
generate_payload = {
    "model": "deepseek-r1:7b",   # 这里需要根据实际情况进行修改
    "prompt": "请生成一个关于人工智能的简短介绍。",  # 这里需要根据实际情况进行修改
    "stream": False,       # 默认使用的是True，如果设置为False，则返回的是一个完整的响应，而不是一个流式响应
    "options": {
       # "num_ctx": 7,
        "num_predict": 512
    }
}

# 调用生成接口
response_generate = requests.post(generate_url, json=generate_payload)
if response_generate.status_code == 200:
    generate_response = response_generate.json()
    print("生成响应:", json.dumps(generate_response, ensure_ascii=False, indent=2))
    generate_response["response"]
    # %%
    # 提取 <think> 标签中的内容
    think_start = generate_response["response"].find("<think>")
    think_end = generate_response["response"].find("</think>")

    if think_start != -1 and think_end != -1:
        think_content = generate_response["response"][think_start + len("<think>"):think_end].strip()
    else:
        think_content = "No think content found."

    # 提取正常的文本内容
    normal_content = generate_response["response"][think_end + len("</think>"):].strip()

    # 打印结果
    print("思考内容:\n", think_content)
    print("\n正常内容:\n", normal_content)
    # 将纳秒转换为秒
    total_duration_s = generate_response["total_duration"] / 1_000_000_000
    load_duration_s = generate_response["load_duration"] / 1_000_000_000
    prompt_eval_duration_s = generate_response["prompt_eval_duration"] / 1_000_000_000
    eval_duration_s = generate_response["eval_duration"] / 1_000_000_000

    # 打印转换后的秒值
    print("单次调用总花费时间:", total_duration_s)
    print("加载模型花费时间:", load_duration_s)
    print("评估提示所花费的时间:", prompt_eval_duration_s)
    print("生成响应的时间:", eval_duration_s)

else:
    print("生成请求失败:", response_generate.status_code, response_generate.text)