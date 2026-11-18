import torch
import torch.nn.functional as F

# 1. 构造输入数据 (3 个 Token，每个维度是 3)
q = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
k = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
v = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])

# 2. 计算 Attention 核心公式： Softmax(Q * K^T / sqrt(d_k)) * V
d_k = q.size(-1)

# 计算相似度
scores = torch.matmul(q, k.transpose(-2, -1))

# 缩放
scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# 转换为概率权重
attention_weights = F.softmax(scaled_scores, dim=-1)

# 加权求和
output = torch.matmul(attention_weights, v)

# 3. 打印结果
print("=== 注意力权重矩阵 ===")
print(attention_weights)
print("\n=== 最终输出 ===")
print(output)