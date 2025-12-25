import json
import argparse


def convert_to_alpaca_format(input_file, output_file):
    # 读取输入 JSON 文件
    with open(input_file, 'r') as file:
        data = json.load(file)

    # 初始化一个空列表来存储转换后的对话
    alpaca_format = []

    # 遍历每个条目（在这个例子中可能有多个条目）
    for entry in data:
        # 将当前条目转换为 Alpaca 格式
        alpaca_entry = {
            "instruction": entry["question"],
            "input": entry["evidence"],
            "output": entry["SQL"]
        }

        # 将转换后的条目添加到最终的格式中
        alpaca_format.append(alpaca_entry)

    # 将转换后的数据写入输出 JSON 文件
    with open(output_file, 'w') as output_file_handle:
        json.dump(alpaca_format, output_file_handle, indent=4)


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Convert CoSQL train data to Alpaca format.")

    # 添加输入文件路径参数
    parser.add_argument("input_file", type=str,
                        help="Path to the input JSON file (e.g., cosql_train.json)")

    # 添加输出文件路径参数
    parser.add_argument("output_file", type=str,
                        help="Path to the output JSON file (e.g., alpaca_cosql_train.json)")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数
    convert_to_alpaca_format(args.input_file, args.output_file)