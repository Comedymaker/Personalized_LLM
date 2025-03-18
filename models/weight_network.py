import torch
import torch.nn as nn

class WeightNetwork(nn.Module):
    def __init__(self, logit_dim=10, hidden_dims=[512, 16]):
        super().__init__()
        self.logit_dim = logit_dim
        
        # 三层网络结构
        self.layer_stack = nn.Sequential(
            nn.Linear(2*logit_dim, hidden_dims[0]),  # 输入层：大模型和小模型的top10 logits拼接
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 2),  # 输出层：两个权重值
            nn.Softmax(dim=-1)  # 使用Softmax激活函数
        )
        
        # 初始化策略
        self._init_weights()
    
    def _init_weights(self):
        for idx, layer in enumerate(self.layer_stack):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                # 对输出层的偏置进行初始化，接近均衡
                if layer == self.layer_stack[4]:
                    layer.bias.data = torch.tensor([0.5, 0.5])
                else:
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, llm_logits, slm_logits):
        # 截取top-k logits (k=10)
        topk_llm = llm_logits.topk(self.logit_dim, dim=-1).values
        topk_slm = slm_logits.topk(self.logit_dim, dim=-1).values

        # 拼接特征
        combined = torch.cat([topk_llm, topk_slm], dim=-1)

        # 生成原始权重
        raw_weights = self.layer_stack(combined)

        # 打印 raw_weights 数据
        print(f"raw_weights: {raw_weights}")

        # 归一化权重
        normalized_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
        
        return normalized_weights  # [weight_llm, weight_slm]
