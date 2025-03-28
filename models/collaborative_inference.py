import torch
from transformers import AutoTokenizer
from utils.config_loader import load_config

class CollaborativeInference:
    def __init__(self, large_model, small_model, weight_network, tokenizer, device):
        self.config = load_config()
        self.large_model = large_model
        self.small_model = small_model
        self.weight_network = weight_network.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_id = tokenizer.eos_token_id  # 获取结束符ID

    def get_logits(self, input_ids, attention_mask):
        # 获取大模型logits
        with torch.no_grad():
            large_out = self.large_model(input_ids=input_ids, attention_mask=attention_mask)
            llm_logits = large_out.logits

        # 获取小模型logits
        with torch.no_grad():
            small_out = self.small_model(input_ids=input_ids, attention_mask=attention_mask)
            slm_logits = small_out.logits
        
        return llm_logits, slm_logits
    
    def forward(self, input_ids, attention_mask, labels=None):
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

        logits_sequence = []
        for step in range(max_length):
            # 跳过已完成的样本
            active = ~finished
            if not active.any():
                break
            
            # print(f"current mask: {current_mask[:1]}")
            llm_logits, slm_logits = self.get_logits(current_ids, current_mask)

            llm_logits = llm_logits.to(torch.float32)
            slm_logits = slm_logits.to(torch.float32)

            # 计算权重和融合 logits
            weights = self.weight_network(llm_logits, slm_logits)
            
            # 新逻辑：强制设置权重为 LLM=0, SLM=1
            # batch_size, seq_len, vocab_size = llm_logits.size()
            # weights = torch.zeros((batch_size, seq_len, 2), device=self.device, dtype=torch.float32)
            # weights[..., 1] = 1.0  # 小模型（SLM）的权重设为1

            weights_llm = weights[:, :, 0].unsqueeze(-1)  # [4,340,1]
            weights_slm = weights[:, :, 1].unsqueeze(-1)  # [4,340,1]

            # 动态维度扩展（利用广播机制）
            combined_logits = (weights_llm * llm_logits) + (weights_slm * slm_logits)

            # 选择下一个 token
            if labels is not None:  # 训练时使用教师强制
                next_token = labels[:, step]  # 使用真实标签作为下一个 token

            else:
                next_logits = combined_logits

                next_logits /= temperature
                if top_k > 0:
                    top_logits, top_indices = torch.topk(next_logits[:, -1, :], top_k)
                    next_logits = torch.full_like(next_logits[:, -1, :], -float("Inf"))
                    next_logits.scatter_(1, top_indices, top_logits)

                probs = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # 更新结束状态
            finished = finished | (next_token == self.eos_token_id)

            # 拼接新 token 到所有样本
            next_token_all = next_token.unsqueeze(1)

            current_ids = torch.cat([current_ids, next_token_all], dim=1)

            current_labels = self.tokenizer.batch_decode(current_ids)

            # print(f"current text: {current_labels[:1]}")

            # 正确更新 current_mask（关键修改）
            new_attention = torch.where(
                finished.unsqueeze(1),  # 如果已结束，设为 0
                torch.tensor(0, dtype=torch.long, device=self.device),
                torch.tensor(1, dtype=torch.long, device=self.device)
            ).expand(batch_size, 1)  # 扩展为 [batch_size, 1]

            current_mask = torch.cat([current_mask, new_attention], dim=1)

            # 填充 logits_sequence，确保每一步的大小一致
            padded_combined_logits = torch.zeros(batch_size, 1, combined_logits.size(-1), dtype=torch.float16 ,device=self.device)
            padded_combined_logits = combined_logits[:, -1, :].unsqueeze(1)

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
        combined_logits = torch.stack(logits_sequence, dim=1).squeeze(2)

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

    def train_forward(self, input_ids, labels):
        # 训练阶段生成过程
        batch_size = input_ids.size(0)
        current_ids = input_ids
        max_length = self.config["base"]["max_length"]
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

        logits_sequence = []
        for step in range(max_length):
            # 跳过已完成的样本
            active = ~finished
            if not active.any():
                break

            llm_logits, slm_logits = self.get_logits(current_ids)

            llm_logits = llm_logits.to(torch.float32)
            slm_logits = slm_logits.to(torch.float32)

            # 计算权重和融合 logits
            weights = self.weight_network(llm_logits, slm_logits)

            weights_llm = weights[:, :, 0].unsqueeze(-1)  # [4,340,1]
            weights_slm = weights[:, :, 1].unsqueeze(-1)  # [4,340,1]

            # 动态维度扩展（利用广播机制）
            combined_logits = (weights_llm * llm_logits) + (weights_slm * slm_logits)

            # 选择下一个 token
            if labels is not None:  # 训练时使用教师强制
                next_token = labels[:, step]  # 使用真实标签作为下一个 token
            else:  # 推理时使用模型生成的 token
                next_token = torch.argmax(combined_logits[:, -1, :], dim=-1)

            # 更新结束状态
            finished = (next_token == self.eos_token_id)

            # 拼接新 token 到所有样本
            next_token_all = torch.full((batch_size, 1), self.eos_token_id, 
                                        dtype=torch.long, device=self.device)
            next_token_all= next_token.unsqueeze(1)

            current_ids = torch.cat([current_ids, next_token_all], dim=1)

            # 填充 logits_sequence，确保每一步的大小一致
            padded_combined_logits = torch.zeros(batch_size, 1, combined_logits.size(-1), dtype=torch.float16 ,device=self.device)
            padded_combined_logits = combined_logits[:, -1, :].unsqueeze(1)

            logits_sequence.append(padded_combined_logits)
            
        
            # 补全剩余的 logits（如果生成提前结束）
        if len(logits_sequence) < max_length:
            pad_logits = torch.zeros(
                batch_size, 1, combined_logits.size(-1),
                dtype=torch.float32,
                device=self.device
            )
            for _ in range(max_length - len(logits_sequence)):
                logits_sequence.append(pad_logits)

        # 使用 torch.stack 来堆叠所有的 logits
        combined_logits = torch.stack(logits_sequence, dim=1).squeeze(2)

        # 确保 output_ids 的长度为 max_length
        if current_ids.size(1) < max_length:
            pad_length = max_length - current_ids.size(1)
            pad_tokens = torch.full(
                (batch_size, pad_length),
                self.eos_token_id,
                dtype=torch.long,
                device=self.device
            )
            current_ids = torch.cat([current_ids, pad_tokens], dim=1)
        else:
            current_ids = current_ids[:, :max_length]
        
        return {
            "output_ids": current_ids,
            "combined_logits": combined_logits
        }

    def generate(self, text):
        # 文本转token
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        # 生成输出
        outputs = self.forward(inputs)

        output_ids = outputs["output_ids"]
        
        # 转回文本
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
