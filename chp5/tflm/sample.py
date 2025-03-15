import torch
from torch.nn import functional as F
from utils import load_pretrained, save_pretrained, BOS_TOKEN, EOS_TOKEN
from .model import Transformer

@torch.no_grad()
def sample(model, vocab, x, steps, temperature=1.0):
    """
    接收一个输入序列 x （形状为 (b, t)）并预测序列中的下一个词元，每次将预测结果反馈给模型。
    用temperature配合随机采样可以增加/减少随机性
    """

    # 设置为评估模式
    model.eval()

    # 生成符合目标长度的序列
    for k in range(steps):
        # 如果对于Transformer, 如果上文过长, 截取前context_size个token
        if x.size(1) >= model.config.context_size:
            x_cond = x[:, -model.config.context_size:]
        # 如果上文不够长，在其末尾进行padding，由于掩码机制，这部分内容不会影响结果
        else:
            pad = torch.zeros(x.size(0), model.config.context_size - x.size(1))
            x_cond = torch.cat((pad.long().to(x.device), x), dim=1)

        # 用模型进行预测
        logits = model(x_cond)
        # Transformer的输出是logit，loss，并且要取第input_length个数据的结果
        input_length = min(x_cond.size(1), model.config.context_size)
        logits = logits[0][:, input_length - 1, :]
        # 提取最后一步的输出结果并按温度缩放，温度越高，采样越随机
        probs = F.softmax(logits / temperature, dim=-1)

        # 根据prob进行多项式采样
        ix = torch.multinomial(probs, num_samples=1)
        if ix == vocab[EOS_TOKEN]:
            break

        # 将结果添加到序列并继续
        x = torch.cat((x, ix), dim=1)
    return x

def sample_tflm(context, steps=10, model_path="tflm.model", temperature=1.0):
    # 判断是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型和词表到可用的设备上
    vocab, model = load_pretrained(model_path, map_location=device)
    # 将context全部小写化并按空格分割
    context = context.lower().split()
    context = model.config.context_size * [BOS_TOKEN] + context

    # 将输入内容转换为id序列
    x = torch.tensor([vocab.convert_tokens_to_ids(context)]).to(device)

    # 生成结果并转换为token序列
    y = sample(model, vocab, x, steps=steps, temperature=temperature)[0]
    y = vocab.convert_ids_to_tokens(y)

    print(" ".join(y))
