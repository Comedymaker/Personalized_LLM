import torch
import os
from transformers import AutoTokenizer, pipeline
from time import perf_counter

#选择设备
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备微调后模型的路径
model_id_final = "results/models/20250305_083105_TinyLlama-1.1B-Chat-v1.0"  # 这里使用你本地保存的微调后模型路径
tokenizer = AutoTokenizer.from_pretrained(model_id_final)

# 准备推理管道
pipe = pipeline(
    "text-generation",
    model=model_id_final,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map=device
)

# 格式化提示语
def formatted_prompt(question: str) -> str:
    return (
        "[INSTRUCTION] Generate a concise academic paper title based on the abstract.\n"
        f"[ABSTRACT] {question}\n"
        f"[TITLE] "
    )


# 测试推理
start_time = perf_counter()

# 你的测试问题或摘要
prompt = formatted_prompt("Stochastic computing is a novel approach to real arithmetic, offering better error tolerance and lower hardware costs over the conventional implementations. Stochastic modules are digital systems that process random bit streams representing real values in the unit interval. Stochastic modules based on finite state machines (FSMs) have been shown to realize complicated arithmetic functions much more efficiently than combinational stochastic modules. However, a general approach to synthesize FSMs for realizing arbitrary functions has been elusive. We describe a systematic procedure to design FSMs that implement arbitrary real-valued functions in the unit interval using the Taylor series approximation.")

# 使用生成管道进行推理
sequences = pipe(
    prompt,
    do_sample=True,
    temperature=0.7,  # 控制生成的随机性
    top_p=0.9,  # 使用 top-p 策略
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=50  # 限制生成的最大长度
)

# 输出推理结果
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# 输出推理时间
output_time = perf_counter() - start_time
print(f"Time taken for inference: {round(output_time, 2)} seconds")
