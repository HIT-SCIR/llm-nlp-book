import random
from collections import defaultdict
from nltk.corpus import reuters

# 以trigram语言模型为例
n = 3

# 存储每个ngram的出现频次
ngram_count = defaultdict(int)
# 存储每个ngram的前缀出现频次
ngram_precedings_count = defaultdict(int)
# 存储每个ngram的前缀所对应的下一个词的列表及每个词出现的概率列表
ngram_prob = {}

# 获取句子中所有的ngram的列表及其前缀列表
def get_ngrams(sentence, n):
    # 在句子前后加上开始符号和结束符号
    sentence = (n - 1) * ['<bos>'] + sentence + ['<eos>']
    ngrams = []
    precedings = []
    for i in range(n - 1, len(sentence)):
        prec = tuple(sentence[i - n + 1:i])
        ngram = tuple((prec, sentence[i]))
        precedings.append(prec)
        ngrams.append(ngram)

    return ngrams, precedings

# 构建ngram及其前缀的出现频次
def build_ngrams_precedings(text):
    for sentence in text:
        ngrams, precedings = get_ngrams(sentence, n)
        for i in range(len(ngrams)):
            ngram = ngrams[i]
            prec = precedings[i]
            ngram_count[ngram] += 1
            ngram_precedings_count[prec] += 1

# 构建ngram的前缀所对应的下一个词的列表及每个词出现的概率列表
def build_ngram_prob():
    for ngram in ngram_count.keys():
        prec, next = ngram
        prob = ngram_count[ngram] / ngram_precedings_count[prec]
        if prec in ngram_prob:
            ngram_prob[prec]['next'].append(next)
            ngram_prob[prec]['prob'].append(prob)
        else:
            ngram_prob[prec] = {'next': [next], 'prob': [prob]}

# 构建语言模型
def build_lm():
    text = reuters.sents()
    build_ngrams_precedings(text)
    build_ngram_prob()

# 生成句子
def generate(length=10):
    word_list = (n - 1) * ['<bos>']
    for _ in range(length):
        try:
            prec = tuple(word_list[1 - n:])
            next_choice = ngram_prob[prec]
            # 从下一个词的列表中根据概率随机选择一个词
            generated_word = random.choices(next_choice['next'], next_choice['prob'])[0]
            word_list.append(generated_word)
        except:
            break
            
    return word_list

build_lm()
word_list = generate(50)
print(f'Word count: {len(word_list)}')
print(f'Generated sentence: {" ".join(word_list)}')
