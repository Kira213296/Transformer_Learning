import torch
import torch.nn as nn

# 1. 设定参数
embed_dim = 8  # 词向量维度
num_heads = 2  # 头数

# 2. 初始化多头注意力模块
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

# 3. 构造输入数据 (seq_len=3, batch_size=1, embed_dim=8)
x = torch.randn(3, 1, 8)

# 4. 计算多头注意力
attn_output, attn_output_weights = multihead_attn(x, x, x)

# 5. 打印结果
print("=== 多头注意力输出 ===")
print(attn_output)
print("\n=== 注意力权重矩阵 ===")
print(attn_output_weights)