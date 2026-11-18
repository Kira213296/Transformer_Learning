import torch
import torch.nn as nn
import math

# 1. 定义位置编码模块 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入上
        return x + self.pe[:x.size(0), :]

# 2. 搭建完整的 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        # 线性映射层，把输入维度转成模型维度
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 核心：Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 最后的输出层
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# 3. 实例化并测试模型
if __name__ == "__main__":
    # 参数：输入特征维度=3，模型维度=8，多头数量=2，编码器层数=2
    model = SimpleTransformer(input_dim=3, d_model=8, nhead=2, num_layers=2)
    
    # 构造输入数据 (序列长度=5, Batch_size=1, 输入维度=3)
    dummy_input = torch.randn(5, 1, 3)
    
    # 前向传播
    output = model(dummy_input)
    
    print("=== 模型结构 ===")
    print(model)
    print("\n=== 输入形状 ===", dummy_input.shape)
    print("=== 输出形状 ===", output.shape)
    print("\n=== 输出结果 ===")
    print(output)