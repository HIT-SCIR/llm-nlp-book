from dataclasses import dataclass
 
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    batch_size: int = 2
    seq_len: int = 3
    n_embd: int = 4
    n_head: int = 2
    n_layer: int = 2

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.n_embd, config.n_embd * 3)

    def forward(self, x):
        B, T, C = x.size() # batch_size, seq_len, n_embd

        # 获得batch中每个输入的q, k, v，并将q, k, v分解为n_head组
        q, k, v = self.proj(x).chunk(3, dim=-1)
        k = k.view(B, T, self.config.n_head, -1).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, -1).transpose(1, 2)

        # 计算自注意力：
        # (B, n_head, T, hs) x (B, n_head, hs, T) -> (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1  = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

if __name__ == '__main__':
    config = Config()
    x = torch.randn(config.batch_size, config.seq_len, config.n_embd)
    self_attn = Transformer(config)
    y = self_attn(x)
    print(y, y.shape)
