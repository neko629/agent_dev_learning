import json
import argparse


def convert_to_sharegpt_format(input_file, output_file):
    # 读取输入 JSON 文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 初始化一个空列表来存储转换后的对话
    sharegpt_format = []

    # 遍历每个条目（在这个例子中可能有多个条目）
    for entry in data:
        # 初始化一个空列表来存储当前条目的对话
        conversation = []

        # 将 final 中的对话添加到对话中
        if "final" in entry:
            conversation.append({
                "from": "human",
                "value": entry["final"]["utterance"]
            })
            conversation.append({
                "from": "gpt",
                "value": entry["final"]["query"]
            })

        # 遍历每个交互
        for interaction in entry["interaction"]:
            # 将用户的指令添加到对话中
            conversation.append({
                "from": "human",
                "value": interaction["utterance"]
            })

            # 将模型的响应添加到对话中
            conversation.append({
                "from": "gpt",
                "value": interaction["query"]
            })

        # 将当前对话添加到最终的格式中
        sharegpt_format.append({
            "conversations": conversation
        })

    # 将转换后的数据写入输出 JSON 文件
    with open(output_file, 'w') as output_file_handle:
        json.dump(sharegpt_format, output_file_handle, indent=4)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Convert CoSQL train data to ShareGPT format.")

    # 添加输入文件路径参数
    parser.add_argument("input_file", type=str,
                        help="Path to the input JSON file (e.g., cosql_train.json)")

    # 添加输出文件路径参数
    parser.add_argument("output_file", type=str,
                        help="Path to the output JSON file (e.g., sharegpt_cosql_train.json)")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数
    convert_to_sharegpt_format(args.input_file, args.output_file)