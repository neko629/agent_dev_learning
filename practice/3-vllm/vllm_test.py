# python
import os
import traceback
from vllm import LLM, SamplingParams

# 可选：提高 vLLM 日志级别以获取更多输出
os.environ.setdefault("VLLM_LOG_LEVEL", "DEBUG")

def main():
    try:
        llm = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            gpu_memory_utilization=0.8,  # 降低显存占用再试
            disable_log_stats=True
        )

        # 2. 定义采样参数
        # max_tokens: 允许生成的最大长度（根据你的显存和需求设置，比如 1024 或 2048）
        # temperature: 控制随机性，0.7 是常用的对话设置
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        prompts = ["请生成一个关于人工智能的简短介绍。包含起源,当前应用,未来发展三个部分，大约200字左右。"]
        outputs = llm.generate(prompts, sampling_params)
        print(outputs)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}\nGenerated Text: {generated_text}")

    except Exception:
        print("LLM 运行发生异常，打印堆栈以便排查：")
        traceback.print_exc()

if __name__ == "__main__":
    main()
