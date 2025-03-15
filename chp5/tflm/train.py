import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from .dataset import TransformerDataset
from .model import Transformer, Config
from utils import load_reuters, save_pretrained, device, get_loader

from tqdm.auto import tqdm

def train_tflm(batch_size, num_epoch):
    corpus, vocab = load_reuters()
    # 设置参数
    train_config = Config(
        vocab_size=len(vocab),
        context_size=64,
        n_embd=128,
        n_head=4,
        n_layer=4)

    dataset = TransformerDataset(corpus, vocab)
    data_loader = get_loader(dataset, batch_size)

    # 负对数似然损失函数，忽略pad_token处的损失
    nll_loss = nn.NLLLoss()
    # 构建TransformerLM，并加载至device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(train_config)
    model.to(device)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            # 生成并计算损失
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss:.2f}")

    save_pretrained(vocab, model, "tflm.model")
