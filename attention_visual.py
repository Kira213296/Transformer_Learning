import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 构造一个单头注意力来演示
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights

# 2. 模拟输入数据 (8个单词/Token，特征维度为16)
seq_len = 8
d_model = 16
x = torch.randn(seq_len, d_model)

# 3. 获取注意力权重
attention = SimpleAttention(d_model)
weights = attention(x).detach().numpy()

# 4. 画热力图
plt.figure(figsize=(8, 6))
plt.imshow(weights, cmap='viridis')
plt.colorbar(label='Attention Weight')
plt.title("Attention Weights Visualization")
plt.xlabel("Key (Source)")
plt.ylabel("Query (Target)")
plt.xticks(np.arange(seq_len))
plt.yticks(np.arange(seq_len))
plt.show()