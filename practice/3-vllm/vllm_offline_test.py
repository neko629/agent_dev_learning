import os
import glob
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
# 你提供的缓存根路径
CACHE_ROOT = "/home/neko/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"

# 显存占用比例（根据你之前的报错，设为 0.8 比较稳妥）
GPU_UTILIZATION = 0.8


# ===========================================

def get_real_model_path(base_path):
    """
    HuggingFace 的缓存目录结构通常是: models--xxx -> snapshots -> <commit_hash> -> model_files
    这个函数自动找到 snapshots 下的第一个文件夹作为真实模型路径
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到路径: {base_path}")

    # 检查是否直接就是模型目录（包含 config.json）
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path

    # 检查 snapshots 目录
    snapshot_pattern = os.path.join(base_path, "snapshots", "*")
    found_dirs = glob.glob(snapshot_pattern)

    if not found_dirs:
        raise FileNotFoundError(
            f"在 {base_path} 下找不到 snapshots 文件夹或该文件夹为空。")

    # 通常取最新的一个（或唯一的那个）
    real_path = found_dirs[0]
    print(f"✅ 定位到真实模型路径: {real_path}")
    return real_path


def main():
    # 1. 获取包含 config.json 的真实路径
    try:
        model_path = get_real_model_path(CACHE_ROOT)
    except Exception as e:
        print(f"❌ 路径错误: {e}")
        return

    # 2. 初始化 vLLM
    # 注意：直接传入绝对路径，vLLM 就会把它当作本地模型，不会尝试联网下载
    print("🚀 正在加载模型，请稍候...")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=GPU_UTILIZATION,
        trust_remote_code=True,  # 某些特定模型需要这个
        max_model_len=2048,  # 限制一下最大长度，防止显存溢出
    )

    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,  # 温度：0.7 比较平衡，创造性与准确性兼顾
        top_p=0.95,
        max_tokens=512
    )

    # 4. 准备 Prompt
    # DeepSeek/Qwen 系列通常建议使用 Chat 格式，这里手动拼接一个简单的对话格式
    prompts = [
        "介绍一下你自己",
        "介绍一下美国的历史",
        "解释一下量子计算的基本原理"
    ]

    # 5. 执行推理
    print("🤖 开始推理...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. 打印结果
    print("\n" + "=" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"【提问】: {prompt.strip().split('user')[-1].replace('<|im_end|>', '').strip()}")
        print(f"【回答】: {generated_text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
