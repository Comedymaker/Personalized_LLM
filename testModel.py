import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.tokenizer import Tokenizer

# 选择设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载模型和分词器
model_name = "results/models/20250327_032710_TinyLlama-1.1B-Chat-v1.0_merged"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = Tokenizer.load_tokenizer()
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()


# 输入 prompt
messages = [
    {
        "role": "system",
        "content": "You are an academic assistant. Generate **ONLY THE TITLE** of a paper based on the abstract below. The title should: 1) Output **only the title** (no explanations, formatting, or extra text); 2) capture the core innovation; 3) include key technical terms; 4) be under 20 words.",
    },
    {"role": "user", "content": "The fact that instructions in programs often produce repetitive results has motivated researchers to explore various techniques, such as value prediction and value reuse, to exploit this behavior. Value prediction improves the available Instruction-Level Parallelism (ILP) in superscalar processors by allowing dependent instructions to be executed speculatively after predicting the values of their input operands. Value reuse, on the other hand, tries to eliminate redundant computation by storing the previously produced results of instructions and skipping the execution of redundant instructions. Previous value reuse mechanisms use a single instruction or a naturally formed instruction group, such as a basic block, a trace, or a function, as the reuse unit. These naturally-formed instruction groups are readily identifiable by the hardware at runtime without compiler assistance. However, the performance potential of a value reuse mechanism depends on its reuse detection time, the number of reuse opportunities, and the amount of work saved by skipping each reuse unit. Since larger instruction groups typically have fewer reuse opportunities than smaller groups, but they provide greater benefit for each reuse-detection process, it is very important to find the balance point that provides the largest overall performance gain. In this paper, we propose a new mechanism called subblock reuse. Subblocks are created by slicing basic blocks either dynamically or with compiler guidance. The dynamic approaches use the number of instructions, numbers of inputs and outputs, or the presence of store instructions to determine the subblock boundaries. The compiler-assisted approach slices basic blocks using data-flow considerations to balance the reuse granularity and the number of reuse opportunities. The results show that subblocks, which can produce up to 36 percent speedup if reused properly, are better candidates for reuse units than basic blocks. Although subblock reuse with compiler assistance has a substantial and consistent potential to improve the performance of superscalar processors, this scheme is not always the best performer. Subblocks restricted to two consecutive instructions demonstrate surprisingly good performance potential as well."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


input_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids

# 初始化生成序列
generated_tokens = input_ids.clone().to(device)

#debug开始
# temperature = 0.1
# top_k = 50

# # current_ids = input_ids.clone().to(device)

# current_ids = torch.tensor([[32000, 32000,
#          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#          32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#          32000, 32000, 32000,    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
#            526,   385, 21567, 20255,  1058,  2337,  5706,   263,  3022,   895,
#            322, 16232,  5650,  3611,  2729,   373,   278,  9846,  4944,   491,
#            278,  1404, 29892,  1728,   738,  7309,   800,   470, 15998, 29889,
#            450,  3611,   881, 29901, 29871, 29896, 29897, 10446,   278,  7136,
#          24233,   362, 29936, 29871, 29906, 29897,  3160,  1820, 16905,  4958,
#          29936, 29871, 29941, 29897,   367,  1090, 29871, 29906, 29900,  3838,
#          29889,    13, 29966,   829, 14816, 29903,  6778,    13,    13,  2887,
#            278,  1353,   310,  1301,   391,   943,   373,   263,  2323, 29830,
#          18172,   304,  6548, 29892,   372,   338,  4100,   304,  1348,  8724,
#            278, 13807, 13501,   310,  6516, 13883,   363, 25871,  8450, 24210,
#            322, 25734, 15278,  2228, 10340,   304, 11157,  4180, 29889,   910,
#           2323,  7097,   287,  8225,  1904, 13071,  1438, 13501,   304, 16035,
#          11407,   871,   278, 13774,  2319,  5253,   310, 15278, 29899,  5563,
#           8943,  1608,  3625,   297,  2280, 11104, 29889,  5806,  3990,  1218,
#            385,  4152,  6674,   307,   985,   272, 11480,   263,  2323, 29830,
#            338, 28326,  1821, 29892,   445, 11258,   338,  9078,   304, 16035,
#          11407,   871, 13774,  1302,  7989, 29899,  3874,  1312,  8943,  1608,
#          29889,  1334, 16193,   263, 21984,  1773,   389,   949,   287, 11258,
#          29892,  2000,   278,  2428,  7097,   287, 11258, 29892,   408,   385,
#           8671, 29889,  1094,   263,  7498, 19515,   310,   263,  9377, 29899,
#          15118,   480,  6774,  1052,   279, 21433,   322,   263,  6674,   307,
#            985,   272, 29899,   265, 29899, 29874, 29899,   305,   666, 29892,
#            445,   716, 21984,  1773,   389, 19715, 11258,   508,   454, 19698,
#            278,  1900,   310,  5923,   322,  5434,  8943, 12837,   322, 14835,
#           5722, 11763, 29889,  2648, 29299,  6516, 29899, 11851,   287,  3244,
#          29899,  5563,  1580,  2785,   363,  2761,   322,   848,  8839,  2063,
#            411,  1065, 29899,  2230,  8454,   310,   848,  8839,  2063, 29892,
#            278,  2428,  7097,   287, 11258,   508, 16035,   277,   278,  2999,
#           3803,  1070,  1907,   310,  8943,  1608,  3625,   297,  2498, 29899,
#          15503,  4220,  2280, 11104,   304, 10032,   278,  8225,   931,   310,
#            263,  2323,  1824, 29889,   518, 29914, 25580, 29962]]).to(device)

# print(f"current ids: {current_ids}")
# print(f"model pad token: {model.config.pad_token_id}")
# attention_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1 ]]).to(device)

# assert current_ids.shape == attention_mask.shape, "Shape mismatch between input_ids and attention_mask"


# with torch.no_grad():
#     small_out = model(input_ids=current_ids, attention_mask=attention_mask)
#     combined_logits = small_out.logits

# next_logits = combined_logits

# next_logits /= temperature
# if top_k > 0:
#     top_logits, top_indices = torch.topk(next_logits[:, -1, :], top_k)
#     next_logits = torch.full_like(next_logits[:, -1, :], -float("Inf"))
#     next_logits.scatter_(1, top_indices, top_logits)

# probs = torch.nn.functional.softmax(next_logits, dim=-1)
# next_token = torch.multinomial(probs, num_samples=1)

# current_ids = torch.cat([current_ids, next_token], dim=1)

# current_ids = current_ids.squeeze(0)

# current_labels = tokenizer.decode(current_ids.tolist())

# print(f"current text: {current_labels}")

#debug结束

# 生成参数
max_length = 50
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

        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 检查结束条件
        if next_token.item() == eos_token_id:
            break

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

# 解码结果
generated_text = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

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
