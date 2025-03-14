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

    def get_logits(self, input_ids):
        # 获取大模型logits
        with torch.no_grad():
            large_out = self.large_model(input_ids=input_ids)
            llm_logits = large_out.logits

        # 获取小模型logits
        with torch.no_grad():
            small_out = self.small_model(input_ids=input_ids)
            slm_logits = small_out.logits
        
        return llm_logits, slm_logits

    def forward(self, input_ids):
        # 生成过程
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

            # 获取活跃样本的 logits
            active_ids = current_ids[active]
            llm_logits, slm_logits = self.get_logits(active_ids)

            # 计算权重和融合 logits
            weights = self.weight_network(llm_logits, slm_logits)

            weights_llm = weights[:, :, 0].unsqueeze(-1)  # [4,340,1]
            weights_slm = weights[:, :, 1].unsqueeze(-1)  # [4,340,1]

            # 动态维度扩展（利用广播机制）
            combined_logits = (weights_llm * llm_logits) + (weights_slm * slm_logits)

            # 选择下一个 token
            next_token = torch.argmax(combined_logits[:, -1, :], dim=-1)

            # 更新结束状态
            finished = (next_token == self.eos_token_id)

            # 拼接新 token 到所有样本
            next_token_all = torch.full((batch_size, 1), self.eos_token_id, 
                                        dtype=torch.long, device=self.device)
            next_token_all = next_token.unsqueeze(1)
            current_ids = torch.cat([current_ids, next_token_all], dim=1)

            # 填充 logits_sequence，确保每一步的大小一致
            padded_combined_logits = torch.zeros(batch_size, 1, combined_logits.size(-1), dtype=torch.float16, device=self.device)
            padded_combined_logits = combined_logits[:, -1, :].unsqueeze(1)

            logits_sequence.append(padded_combined_logits)
        # 使用 torch.stack 来堆叠所有的 logits
        combined_logits = torch.stack(logits_sequence, dim=1)

        return {
            "output_ids": current_ids,
            "combined_logits": combined_logits
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

            # 计算权重和融合 logits
            weights = self.weight_network(llm_logits, slm_logits)
            print(weights)
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
            
        # 使用 torch.stack 来堆叠所有的 logits
        combined_logits = torch.stack(logits_sequence, dim=1).squeeze(2)

        print(combined_logits)

        return {
            "output_ids": current_ids,
            "combined_logits": combined_logits
        }

    def generate(self, text, max_length=64):
        # 文本转token
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        # 生成输出
        output_ids = self.forward(inputs, max_length=max_length)
        
        # 转回文本
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
