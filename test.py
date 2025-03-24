import torch
import os
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from time import perf_counter

# 选择设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备微调后模型的路径
# model_id_final = "results/models/20250320_074810_TinyLlama-1.1B-Chat-v1.0_merged"  # 这里使用你本地保存的微调后模型路径
model_id_final = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id_final)

messages = [
    {
        "role": "system",
        "content": "You are an academic assistant who always generate a concise and accurate paper title based on the abstract provided by the user, without any explanations or formatting. The title should: 1) capture the core innovation; 2) include key technical terms; 3) be under 20 words.",
    },
    {"role": "user", "content": "Address correlation is a technique that links the addresses that reference the same data values. Using a detailed source-code level analysis, a recent study [1] revealed that different addresses containing the same data can often be correlated at run-time to eliminate on-chip data cache misses. In this paper, we study the upper-bound performance of an Address Correlation System (ACS), and discuss specific optimizations for a realistic hardware implementation. An ACS can effectively eliminate most of the L1 data cache misses by supplying the data from a correlated address already found in the cache to thereby improve the performance of the processor. For 10 of the SPEC CPU2000 benchmarks, 57 to 99% of all L1 data cache load misses can be eliminated, which produces an increase of 0 to 243% in the overall performance of a superscalar processor. We also show that an ACS with 1-2 correlations for a value can usually provide comparable performance results to that of the upper bound. Furthermore, a considerable number of correlations can be found within the same set in the L1 data cache, which suggests that a low-cost ACS implementation is possible. "},
    {"role": "assistant", "content": "Improving Data Cache Performance via Address Correlation: An Upper Bound Study"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_special_tokens=True)

print(f"prompt: {prompt}")

input_ids = tokenizer(prompt, return_tensors="pt")

print(f"input_ids: {input_ids['input_ids']}")

token_id = tokenizer("[/INST]")

print(token_id)
