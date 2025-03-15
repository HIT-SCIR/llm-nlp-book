import torch
from torch import nn
from torch.nn import functional as F
from utils import init_weights
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, vocab_size, context_size, n_embd=2, n_head=2, n_layer=2):
        """

        :param vocab_size: 词表大小
        :param context_size: 最大序列长度, 即Transformer块的"大小"
        :param batch_size: 批次大小
        :param n_embd: 词向量维度
        :param n_head: 注意力头数
        :param n_layer: 注意力层数
        """
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.context_size = context_size

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 保存模型配置
        self.config = config

        # 保证n_embd可以被n_head整除
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # 将向量映射到q/k/v
        self.proj = nn.Linear(config.n_embd, config.n_embd * 3)

        # 注意力掩码: 不对当前token之后的内容施加注意力, 避免模型看到未来的信息
        self.register_buffer("mask", torch.tril(torch.ones(config.context_size, config.context_size))
                             .view(1, 1, config.context_size, config.context_size))

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, n_embd

        # 获得batch中每个输入的q, k, v
        # x(batch_size, seq_len, n_embd) --proj--> (batch_size, seq_len, n_embd*3)
        # --chunk--> q,k,v(batch_size, seq_len, n_embd)
        q, k, v = self.proj(x).chunk(3, dim=-1)

        # 将q, k, v分解为n_head组, 每个head对应的向量维度为n_embd/n_head, 在第四维
        k = k.view(B, T, self.config.n_head, -1).transpose(1, 2)
        q = q.view(B, T, self.config.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, -1).transpose(1, 2)

        # 计算自注意力分数
        # (B, n_head, T, hs) x (B, n_head, hs, T) -> (B, n_head, T, T)
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

        # 应用掩码
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # 将注意力分数转化为注意力分布
        attn = F.softmax(attn, dim=-1)

        # 注意力分布与v相乘, 得到注意力输出
        y = attn @ v

        # head组的输出拼接起来
        y = y.transpose(1, 2).reshape(B, T, C)

        return y


class MLP(nn.Module):
    """
    两层全连接网络
    用于为Transformer的每个Block添加非线性表示能力
    """

    def __init__(self, config):
        super().__init__()
        # 隐层, 将向量映射到4倍的维度
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        # 激活
        self.gelu = nn.GELU()
        # 输出层, 将向量映射回原来的维度
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    """
    Transformer的基本单元
    在每个子层的入口进行归一化和残差连接
    """

    def __init__(self, config):
        super().__init__()
        # 归一化
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 多头自注意力块
        self.attn = MultiHeadSelfAttention(config)
        # 归一化
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 前馈网络
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (batch_size, seq_len, n_embd)

        # self.attn(x) 对 x 应用多头自注意力
        # x + self.attn(x)的过程为残差连接
        # self.ln_1对残差连接的结果进行归一化
        x = self.ln_1(x + self.attn(x))

        # 应用前馈网络, 并进行残差连接和归一化
        x = self.ln_2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    """
    Transformer模型
    输入部分: 词向量 + 位置向量 + dropout
    编码部分: 由多个Block组成
    输出部分: 归一化 + 线性映射
    """

    def __init__(self, config):
        super().__init__()
        # 配置信息
        self.config = config

        # 词向量: 将输入的id映射为词向量
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置向量: 将输入的位置映射为位置向量
        self.pos_emb = nn.Embedding(config.context_size, config.n_embd)
        # 层归一化: 对输入进行归一化(块间和块输出已经进行了归一化)
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 编码层: 由多个Transformer块组成
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 解码层: 将输出的词向量映射为词id
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, y=None):
        # 要求输入序列长度不能大于块大小
        _, seq_len = x.size()
        assert seq_len <= self.config.context_size, "Cannot forward, model block size is exhausted."

        # 获取词向量
        # x(batch_size, seq_len) --> token_embeddings: (batch_size, seq_len, n_embd)
        token_embeddings = self.tok_emb(x)

        # 获取位置向量
        pos = torch.arange(seq_len, dtype=torch.long).to(x.device)
        position_embeddings = self.pos_emb(pos)

        # 二者相加作为输入
        x = token_embeddings + position_embeddings

        x = self.ln_f(x)

        # 通过多个Transformer块进行编码
        for block in self.blocks:
            x = block(x)

        # 解码为对下一个token的回归预测
        # x(batch_size, seq_len, n_embd) --> logits(batch_size, seq_len, vocab_size)
        logits = self.head(x)

        # 如果有给定的目标输出, 则计算对数似然损失
        loss = None
        if y is not None:
            # 计算损失
            # x(batch_size, seq_len, vocab_size) --> x(batch_size*seq_len, vocab_size)
            # y(batch_size * seq_len)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss
