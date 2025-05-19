import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.tokenizer import Tokenizer
from models.model import TinyModelLoader, LargeModelLoader


# 选择设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("here")

# 加载模型和分词器
tokenizer = Tokenizer.load_tokenizer()
token_id = 18
decoded_text = tokenizer.decode([token_id], skip_special_tokens=False)
print(decoded_text)

model = TinyModelLoader.load_finetuned_model()
# model = LargeModelLoader.load_model()
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

print(tokenizer.pad_token_id)

# 输入 prompt
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that only outputs a rating from 1 to 5 for the given user review. Do not explain with any word. Only respond with a single number.",
    },
    {"role": "user", "content": "Got this tie set for our valentine's day dinner. It looked great, fit well and the packaging didn't leave any creases the way that some do. The handkerchief was creased heavily however and took a dry cleaning run to come out."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(prompt)

input_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids

# 初始化生成序列
generated_tokens = input_ids.clone().to(device)

allowed_token_ids = []
for i in range(1, 6):
    ids = tokenizer(str(i), add_special_tokens=False).input_ids
    allowed_token_ids.append(ids[-1])  # 取最后一个 token，通常是实际数字
allowed_token_ids = torch.tensor(allowed_token_ids, device=device)

print("Allowed token ids:", allowed_token_ids)
print("Decoded:", [tokenizer.decode([i]) for i in allowed_token_ids])

# 生成参数
max_length = 35
temperature = 0.1
top_k = 50
eos_token_id = tokenizer.eos_token_id

# 开始生成
with torch.no_grad():
    for _ in range(max_length):
        outputs = model(generated_tokens)
        next_token_logits = outputs.logits[:, -1, :]

        # 应用温度和 top-k
        next_token_logits /= temperature
        if top_k > 0:
            top_logits, top_indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, -float("Inf"))
            next_token_logits.scatter_(1, top_indices, top_logits)


        # logits_strict = torch.full_like(next_token_logits, float('-inf'))
        # logits_strict[:, allowed_token_ids] = next_token_logits[:, allowed_token_ids]
        # next_token_logits = logits_strict

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 检查结束条件
        if next_token.item() == eos_token_id:
            break

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

# 解码结果
generated_text = tokenizer.decode(generated_tokens[0][input_ids.shape[-1]:], skip_special_tokens=False).strip()

print("Generated text:")
print(generated_text)

# import torch
# import os
# from transformers import AutoTokenizer, pipeline
# from time import perf_counter

# # 选择设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 准备微调后模型的路径
# # model_id_final = "results/models/20250324_024541_TinyLlama-1.1B-Chat-v1.0_merged"  # 这里使用你本地保存的微调后模型路径
# model_id_final = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
# tokenizer.pad_token_id = pad_token_id


# # 准备推理管道
# pipe = pipeline(
#     "text-generation",
#     model=model_id_final,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map=device
# )

# messages = [
#     {
#         "role": "system",
#         "content": "You are an academic assistant who always generate a concise and accurate paper title based on the abstract provided by the user, without any explanations or formatting. The title should: 1) capture the core innovation; 2) include key technical terms; 3) be under 20 words.",
#     },
#     {"role": "user", "content": "This paper presents a new protocol, Self-tuning ActiveData-aware Cache Consistency (SADCC), which employsparallel communication and self-tuning speculation toimprove the performance of data-shipping database systems.Using parallel communication with simultaneous client-serverand client-client communication, SADCC reduces thenetwork latency for detecting data conflicts by 50%, whileincreasing message volume overhead by only about 4.8%. Bybeing aware of the global states of cached data, clients self-tunebetween optimistic and pessimistic consistency control.The abort rate is reduced by statistically quantifying thespeculation cost. We compare SADCC with the leadingcache consistency algorithms, Active Data-aware CacheConsistency (ADCC) and Asynchronous Avoidance-basedCache Consistency (AACC), in a page server DBMSarchitecture with page-level consistency. The experimentsshow that, in a non-contention environment, both SADCCand ADCC display a slight reduction (an average of 2.3%)in performance compared to AACC with a high-speednetwork environment. With high contention, however,SADCC has an average of 14% higher throughput thanAACC and 6% higher throughput than ADCC."},
# ]

# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# # 测试推理
# start_time = perf_counter()

# # 你的测试问题或摘要
# # prompt = formatted_prompt("Stochastic computing is a novel approach to real arithmetic, offering better error tolerance and lower hardware costs over the conventional implementations. Stochastic modules are digital systems that process random bit streams representing real values in the unit interval. Stochastic modules based on finite state machines (FSMs) have been shown to realize complicated arithmetic functions much more efficiently than combinational stochastic modules. However, a general approach to synthesize FSMs for realizing arbitrary functions has been elusive. We describe a systematic procedure to design FSMs that implement arbitrary real-valued functions in the unit interval using the Taylor series approximation.")

# # 使用生成管道进行推理
# sequences = pipe(
#     prompt,
#     do_sample=True,
#     temperature=0.1,  # 控制生成的随机性
#     top_p=0.9,  # 使用 top-p 策略
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_new_tokens=50  # 限制生成的最大长度
# )

# # 输出推理结果
# for seq in sequences:
#     generated_text = seq['generated_text']
#     print(f"Generated Text: {generated_text}")

#     # 提取生成的回答部分，排除掉原始的 prompt 部分
#     input_tokens = tokenizer(prompt, return_tensors="pt").input_ids
#     generated_tokens = tokenizer(generated_text, return_tensors="pt").input_ids

#     # 从生成的输出中提取模型的回答部分
#     answer_tokens = generated_tokens[0][len(input_tokens[0]):]
    
#     # 解码生成的回答部分
#     answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
#     print(f"Extracted Answer: {answer_text}")

# # 输出推理时间
# output_time = perf_counter() - start_time
# print(f"Time taken for inference: {round(output_time, 2)} seconds")
