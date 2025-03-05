import torch
import torch.nn as nn

class WeightNetwork(nn.Module):
    def __init__(self, input_size):
        super(WeightNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # 输入是两个模型logits的拼接
        self.sigmoid = nn.Sigmoid()  # 输出的权重应该是一个介于0到1之间的值

    def forward(self, logits):
        x = self.fc(logits)
        return self.sigmoid(x)  # 通过sigmoid确保输出在0到1之间
