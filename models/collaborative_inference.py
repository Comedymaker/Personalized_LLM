import torch
from transformers import AutoTokenizer
from utils.config_loader import load_config
from utils.replay_buffer import ReplayBuffer
import torch.nn.functional as F
from utils.compute_conf import compute_confidence_features

class CollaborativeInference:
    def __init__(self, large_model, small_model, weight_network, tokenizer, device):
        self.config = load_config()
        self.large_model = large_model
        self.small_model = small_model
        self.weight_network = weight_network.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = tokenizer.eos_token_id  # 获取结束符ID
        self.replay_buffer = ReplayBuffer(max_size=1000)  # 初始化重放缓冲区
        self.llm_past_key_values = None
        self.slm_past_key_values = None

        self.invalid_token_strings = [
            self.tokenizer.pad_token,
            "<|im_start|>",
            "<|im_end|>",
            "<s>", "</s>",  # 兼容其他模型的特殊符号
        ]
        self.invalid_token_ids = set()
        for token in self.invalid_token_strings:
            ids = self.tokenizer(token, add_special_tokens=False).input_ids
            self.invalid_token_ids.update(ids)

    def get_outputs(self, input_ids, attention_mask, llm_past_key_values=None, slm_past_key_values=None, use_past=True):
        # 获取大模型logits
        with torch.no_grad():
            large_out = self.large_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=use_past, 
                output_hidden_states=False,
                past_key_values=self.llm_past_key_values if use_past else None)

        # 获取小模型logits
        with torch.no_grad():
            small_out = self.small_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                use_cache=use_past, 
                output_hidden_states=True,
                past_key_values=self.slm_past_key_values if use_past else None)
        
        return large_out, small_out
    
    def forward(self, input_ids, attention_mask, labels=None, use_past=True):
        # 训练阶段生成过程
        batch_size = input_ids.size(0)
        current_ids = input_ids
        current_labels = self.tokenizer.batch_decode(current_ids)
        input_length = input_ids.size(1)
        # print(f"input text: {current_labels[:1]}")
        max_length = self.config["base"]["max_length"]
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        top_k = self.config["base"]["top_k"]
        temperature = self.config["base"]["temperature"]
        current_mask = attention_mask

        self.llm_past_key_values = None
        self.slm_past_key_values = None

        logits_sequence = []
        for step in range(max_length):
            # 跳过已完成的样本
            active = ~finished
            if not active.any():
                break
            
            if use_past:
                if self.llm_past_key_values is None:
                    llm_inputs = current_ids
                    llm_masks = current_mask
                else:
                    llm_inputs = current_ids[:, -1:]
                    llm_masks = current_mask[:, -1:]

                if self.slm_past_key_values is None:
                    slm_inputs = current_ids
                    slm_masks = current_mask
                else:
                    slm_inputs = current_ids[:, -1:]
                    slm_masks = current_mask[:, -1:]
                
            else:
                llm_inputs = current_ids
                llm_masks = current_mask


            # decoded_input = self.tokenizer.decode(llm_inputs[0], skip_special_tokens=False)
            # print("Decoded llm_input[0]:", decoded_input)
            llm_outputs, slm_outputs = self.get_outputs(
                input_ids=llm_inputs,
                attention_mask=llm_masks,
                llm_past_key_values=self.llm_past_key_values,
                slm_past_key_values=self.slm_past_key_values,
                use_past=use_past
            )

            # 更新缓存
            self.llm_past_key_values = llm_outputs.past_key_values
            self.slm_past_key_values = slm_outputs.past_key_values

            llm_logits = llm_outputs.logits.to(torch.float32)
            slm_logits = slm_outputs.logits.to(torch.float32)

            llm_last_token = llm_logits[:, -1, :]  # 取最后一个 token
            slm_last_token = slm_logits[:, -1, :]  # 取最后一个 token

            probs_s = F.softmax(slm_last_token, dim=-1)
            probs_l = F.softmax(llm_last_token, dim=-1)

            entropy_s = -torch.sum(probs_s * torch.log(probs_s + 1e-8), dim=-1)  # [B]
            entropy_l = -torch.sum(probs_l * torch.log(probs_l + 1e-8), dim=-1)  # [B]

            last_hidden_state = slm_outputs.hidden_states[-1]  
            slm_hidden_states = last_hidden_state[:, -1, :].to(torch.float32) 

            conf_feat = compute_confidence_features(slm_last_token, llm_last_token, topk=1)  # [B, 5]

            # 计算权重和融合 logits
            # weights = self.weight_network(llm_last_token, slm_last_token)
            weights = self.weight_network(slm_hidden_states, conf_feat)
            
            # print(f"weights: {weights}")
            # print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")    # 当前分配（已用）
            # print(torch.cuda.memory_reserved() / 1024**3, "GB reserved")       # 当前预留（buffer区）
            # print(torch.cuda.max_memory_allocated() / 1024**3, "GB max allocated")  # 运行过程中最大占用
            # print(torch.cuda.max_memory_reserved() / 1024**3, "GB max reserved")

            # weights_llm = weights[:, :, 0].unsqueeze(-1)  # [4,340,1]
            # weights_slm = weights[:, :, 1].unsqueeze(-1)  # [4,340,1]
            weights_llm = weights
            weights_slm = 1 - weights

            # 动态维度扩展（利用广播机制）
            # combined_logits = (weights_llm * llm_logits) + (weights_slm * slm_logits)
            combined_logits = (weights_llm * llm_last_token) + (weights_slm * slm_last_token)

            # debug
            # decoded_input = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
            # decoded_target = self.tokenizer.decode([labels[0, step].item()], skip_special_tokens=False)
                
            # print(f"\n📥 Sample {step + 1}")
            # print(f"[Input IDs] {current_ids[0].tolist()}")
            # print(f"[Decoded Input] {decoded_input}")
            # print(f"[Label Token ID] {labels[0, step].item()} -> Token: {decoded_target}")

            # print(f"Combined logits shape: {combined_logits.shape}")

            # logits = combined_logits[0]  # 取第pos个样本的第 0 个位置
            # probs = F.softmax(logits, dim=-1)
            # topk = torch.topk(probs, k=10)

            # print(f"\n🧠 Token Position {step + 1}:")
            # for i, (token, prob) in enumerate(zip(
            #     self.tokenizer.convert_ids_to_tokens(topk.indices.tolist()),
            #     topk.values.tolist()
            # )):
            #     print(f"Top {i+1}: Token = {token}, Probability = {prob:.4f}")

            # decoded_labels = [self.tokenizer.decode(label_ids, skip_special_tokens=False) for label_ids in labels[0]]
            # print(decoded_labels)

            # llm_probs = torch.nn.functional.softmax(llm_logits, dim=-1)
            # slm_probs = torch.nn.functional.softmax(slm_logits, dim=-1)
            # combined_probs = (weights_llm * llm_probs) + (weights_slm * slm_probs)
            # combined_logits = torch.log(combined_probs + 1e-8)  # 避免 log(0)


            # 选择下一个 token
            if labels is not None:  # 训练时使用教师强制
                next_token = labels[:, step]  # 使用真实标签作为下一个 token

                if self.replay_buffer is not None:
                    # 将当前输入和标签存入重放缓冲区
                    for b in range(batch_size):
                        label_token = labels[b, step].item()
                        if label_token in self.invalid_token_ids:
                            continue  # 跳过特殊符号 token
                        sample = {
                            "input_ids": current_ids[b].detach().cpu(),         # 当前上下文
                            "attention_mask": current_mask[b].detach().cpu(),
                            "label_token_id": labels[b, step].item()
                        }
                        score = abs(entropy_l[b] - entropy_s[b]).item()  # 或 log prob 差值
                        self.replay_buffer.add(sample, score)
                        

            else:
                next_logits = combined_logits

                next_logits /= temperature
                if top_k > 0:
                    top_logits, top_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, -float("Inf"))
                    next_logits.scatter_(1, top_indices, top_logits)

                probs = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # 更新结束状态
            finished = finished | (next_token == self.eos_token_id)

            # 拼接新 token 到所有样本
            next_token_all = next_token.unsqueeze(1)

            current_ids = torch.cat([current_ids, next_token_all], dim=1)

            # current_labels = self.tokenizer.batch_decode(current_ids)

            # print(f"current text: {current_labels[:1]}")

            # 正确更新 current_mask（关键修改）
            new_attention = torch.where(
                finished.unsqueeze(1),  # 如果已结束，设为 0
                torch.tensor(0, dtype=torch.long, device=self.device),
                torch.tensor(1, dtype=torch.long, device=self.device)
            ).expand(batch_size, 1)  # 扩展为 [batch_size, 1]

            current_mask = torch.cat([current_mask, new_attention], dim=1)

            # 填充 logits_sequence，确保每一步的大小一致
            # padded_combined_logits = torch.zeros(batch_size, 1, combined_logits.size(-1), dtype=torch.float16 ,device=self.device)
            # padded_combined_logits = combined_logits[:, -1, :].unsqueeze(1)
            padded_combined_logits = combined_logits.unsqueeze(1)  # [batch_size, 1, vocab_size]
            logits_sequence.append(padded_combined_logits)

            # 补全剩余的 logits（如果生成提前结束）
        total_max_length = input_length + max_length
        if len(logits_sequence) < max_length:
            pad_logits = torch.zeros(
                batch_size, 1, combined_logits.size(-1),
                dtype=torch.float32,
                device=self.device
            )
            for _ in range(max_length - len(logits_sequence)):
                logits_sequence.append(pad_logits)
            
        # 使用 torch.stack 来堆叠所有的 logits
        combined_logits = torch.cat(logits_sequence, dim=1)

        # 确保 output_ids 的长度为 max_length
        if current_ids.size(1) < total_max_length:
            pad_length = total_max_length - current_ids.size(1)
            pad_tokens = torch.full(
                (batch_size, pad_length),
                self.eos_token_id,
                dtype=torch.long,
                device=self.device
            )
            current_ids = torch.cat([current_ids, pad_tokens], dim=1)
        else:
            current_ids = current_ids[:, :total_max_length]

        # 新增：提取生成的 token（不包含输入部分）
        generated_tokens = current_ids[:, input_length:]

        return {
            "output_ids": current_ids,
            "combined_logits": combined_logits,
            "generated_tokens": generated_tokens  # 新增的返回项
        }

    def generate(self, text):
        # 文本转token
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        # 生成输出
        attention_mask = torch.ones_like(inputs)
        outputs = self.forward(inputs, attention_mask)


        output_ids = outputs["output_ids"]
        
        # 转回文本
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
