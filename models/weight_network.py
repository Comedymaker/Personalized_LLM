# ## Logits implement
# import torch
# import torch.nn as nn

# class WeightNetwork(nn.Module):
#     def __init__(self, vocab_size, hidden_dims=[512, 16]):
#         super().__init__()
#         self.logit_dim = 20
        
#         # 三层网络结构
#         self.layer_stack = nn.Sequential(
#             nn.Linear(2*self.logit_dim, hidden_dims[0]),  # 输入层：大模型和小模型的top10 logits拼接
#             nn.ReLU(),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[1], 1),  # 输出层：两个权重值
#             nn.Sigmoid()  # 使用 Sigmoid 激活函数映射到 [0, 1]
#         )
        
#         # 初始化策略
#         self._init_weights()
    
#     def _init_weights(self):
#         for idx, layer in enumerate(self.layer_stack):
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#                 # 对输出层的偏置进行初始化，接近均衡
#                 if layer == self.layer_stack[4]:
#                     layer.bias.data = torch.tensor([0.5])
#                 else:
#                     nn.init.constant_(layer.bias, 0.1)

#     def forward(self, llm_logits, slm_logits):
#         # 截取top-k logits (k=10)
#         topk_llm = llm_logits.topk(self.logit_dim, dim=-1).values
#         topk_slm = slm_logits.topk(self.logit_dim, dim=-1).values

#         # 拼接特征
#         combined = torch.cat([topk_llm, topk_slm], dim=-1)

#         # 生成原始权重
#         raw_weights = self.layer_stack(combined)
        
#         return raw_weights  # [weight_llm, weight_slm]

# entropy implementation

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class WeightNetwork(nn.Module):
#     def __init__(self, vocab_size, hidden_dims=[512, 512]):
#         super().__init__()
        
#         # 输入层的维度为 2 * |V|，即拼接后的维度
#         input_dim = 2
        
#         self.layer_stack = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, hidden_dims[0]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[1], vocab_size),
#             nn.Sigmoid()  # 使用 Sigmoid 激活函数映射到 [0, 1]
#         )

#         self._init_weights()

#     def _init_weights(self):
#         for layer in self.layer_stack:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
#                 nn.init.constant_(layer.bias, 0.1)

#     def forward(self, llm_logits, slm_logits):
#         # 选择最后一个 token 的 logits

#         probs_s = F.softmax(slm_token, dim=-1)
#         probs_l = F.softmax(llm_token, dim=-1)

#         entropy_s = -torch.sum(probs_s * torch.log(probs_s + 1e-8), dim=-1)  # [B]
#         entropy_l = -torch.sum(probs_l * torch.log(probs_l + 1e-8), dim=-1)  # [B]

#         # 拼接 LLM 和 SLM 的最后一个 token 的 logits
#         # combined = torch.cat([llm_token, slm_token], dim=-1)
#         combined = torch.cat([
#             entropy_s.unsqueeze(-1), 
#             entropy_l.unsqueeze(-1)
#         ], dim=-1)  # shape: [B, 2]


#         # 通过网络
#         output = self.layer_stack(combined)  # shape: [batch_size, vocab_size]

#         return output



# CoDi implementation

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dims=[512, 512], ctx_dim=2048, conf_dim=5):
        super().__init__()
        
        # 投影置信度特征
        self.conf_proj = nn.Sequential(
            nn.LayerNorm(conf_dim),
            nn.Linear(conf_dim, 128),
            nn.ReLU()
        )
        
        # 投影上下文 hidden state
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, 512),
            nn.ReLU()
        )
        
        # 主 MLP，输出 logits（或也可输出权重）
        self.mlp = nn.Sequential(
            nn.Linear(512 + 128, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], vocab_size),  # 输出 vocab logits
            nn.Sigmoid()  # 使用 Sigmoid 激活函数映射到 [0, 1]
        )
        
        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, ctx_hidden, conf_feat):
        ctx_emb = self.ctx_proj(ctx_hidden)      # [B, 512]
        conf_emb = self.conf_proj(conf_feat)     # [B, 128]

        x = torch.cat([ctx_emb, conf_emb], dim=-1)  # [B, 640]

        return self.mlp(x)
