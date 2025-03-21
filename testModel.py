import torch
import os
from transformers import AutoTokenizer, pipeline
from time import perf_counter

# 选择设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备微调后模型的路径
model_id_final = "results/models/20250321_083509_TinyLlama-1.1B-Chat-v1.0_merged"  # 这里使用你本地保存的微调后模型路径
# model_id_final = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id_final)

# 准备推理管道
pipe = pipeline(
    "text-generation",
    model=model_id_final,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map=device
)

messages = [
    {
        "role": "system",
        "content": "You are an academic assistant who always generate a concise and accurate paper title based on the abstract provided by the user, without any explanations or formatting. The title should: 1) capture the core innovation; 2) include key technical terms; 3) be under 20 words.",
    },
    {"role": "user", "content": "Value reuse improves a processor's performance by dynamically caching the results of previous instructions and reusing those results to bypass the execution of future instructions that have the same opcode and input operands. However, continually replacing the least recently used entries could eventually fill the value reuse table with instructions that are not frequently executed. Furthermore, the complex hardware that replaces entries and updates the table may necessitate an increase in the clock period. We propose instruction precomputation to address these issues by profiling programs to determine the opcodes and input operands that have the highest frequencies of execution. These instructions then are loaded into the precomputation table before the program executes. During program execution, the precomputation table is used in the same way as the value reuse table is, with the exception that the precomputation table does not dynamically replace any entries. For a 2K-entry pre-computation table implemented on a 4-way issue machine, this approach produced an average speedup of 11.0%. By comparison, a 2K-entry value reuse table produced an average speedup of 6.7%. Instruction precomputation outperforms value reuse, especially for smaller tables, with the same number of table entries while using less area and having a lower access time."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 格式化提示语
def formatted_prompt(question: str) -> str:
    # return (
    #     "[INSTRUCTION] Generate a concise academic paper title based on the abstract.\n"
    #     f"[ABSTRACT] {question}\n"
    #     f"[TITLE] "
    # )
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

# 测试推理
start_time = perf_counter()

# 你的测试问题或摘要
# prompt = formatted_prompt("Stochastic computing is a novel approach to real arithmetic, offering better error tolerance and lower hardware costs over the conventional implementations. Stochastic modules are digital systems that process random bit streams representing real values in the unit interval. Stochastic modules based on finite state machines (FSMs) have been shown to realize complicated arithmetic functions much more efficiently than combinational stochastic modules. However, a general approach to synthesize FSMs for realizing arbitrary functions has been elusive. We describe a systematic procedure to design FSMs that implement arbitrary real-valued functions in the unit interval using the Taylor series approximation.")

# 使用生成管道进行推理
sequences = pipe(
    prompt,
    do_sample=True,
    temperature=0.1,  # 控制生成的随机性
    top_p=0.9,  # 使用 top-p 策略
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=50  # 限制生成的最大长度
)

# 输出推理结果
for seq in sequences:
    generated_text = seq['generated_text']
    print(f"Generated Text: {generated_text}")

    # 提取生成的回答部分，排除掉原始的 prompt 部分
    input_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

    # 从生成的输出中提取模型的回答部分
    answer_tokens = generated_tokens[0][len(input_tokens[0]):]
    
    # 解码生成的回答部分
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    print(f"Extracted Answer: {answer_text}")

# 输出推理时间
output_time = perf_counter() - start_time
print(f"Time taken for inference: {round(output_time, 2)} seconds")
