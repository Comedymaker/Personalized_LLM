import torch
from transformers import AutoTokenizer

class CollaborativeInference:
    def __init__(self, large_model, small_model, weight_network, tokenizer, device):
        self.large_model = large_model
        self.small_model = small_model
        self.weight_network = weight_network
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_ids, max_length=64):
        # 获取大模型和小模型的 logits
        with torch.no_grad():
            large_model_output = self.large_model(input_ids=input_ids, labels=input_ids)
            small_model_output = self.small_model(input_ids=input_ids, labels=input_ids)
        
        large_model_logits = large_model_output.logits
        small_model_logits = small_model_output.logits
        
        # 拼接两个模型的 logits
        logits = torch.cat([large_model_logits, small_model_logits], dim=-1)
        
        # 使用权重网络来计算加权
        weights = self.weight_network(logits)  # 权重是 0 到 1 之间的值
        
        # 通过加权两个模型的 logits
        weighted_logits = weights * large_model_logits + (1 - weights) * small_model_logits
        
        # 使用加权后的 logits 生成最终的输出
        output = self.large_model.generate(input_ids=input_ids, logits=weighted_logits, max_length=max_length)
        return output
