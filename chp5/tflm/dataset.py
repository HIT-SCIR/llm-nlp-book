import torch
from torch.utils.data import Dataset
from utils import BOS_TOKEN, EOS_TOKEN
from tqdm.auto import tqdm

class TransformerDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=16):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # 插入句首句尾符号
            sentence = context_size * [self.bos] + sentence + [self.eos]
            for i in range(context_size, len(sentence)):
                # 模型输入：长为context_size的上文
                context = sentence[i - context_size:i]
                # 模型输出：模型输入的下一个词构成的长为context_size的序列
                target = sentence[i - context_size + 1: i + 1]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)
