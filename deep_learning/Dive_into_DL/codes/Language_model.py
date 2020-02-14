#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: Language_model.py
# @time: 2020/2/14 10:39

# read dataset
path_text = 'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\\' + \
            'Dive_into_DL\materials\Task02\jaychou_lyrics.txt'
with open(path_text, 'r', encoding='utf-8') as f:
    corpus_chars = f.read()
print(len(corpus_chars))
print(corpus_chars[: 40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[: 10000]

# build char2idx
# 去重，得到索引到字符的映射
idx_to_char = list(set(corpus_chars))
# 字符到索引的映射
char_to_idx = {char: i for i, char in enumerate(idx_to_char)}
vocab_size = len(char_to_idx)
print(vocab_size)
# 将每个字符转化为索引，得到一个索引的序列
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[: 20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)


def load_data_jay_lyrics():
    with open('/home/kesci/input/jaychou_lyrics4703/jaychou_lyrics.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 时序数据的采样
import torch
import random


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符，
    # 因为后面要留一个预测的
    # 下取整，得到不重叠情况下的样本个数
    num_examples = (len(corpus_indices) - 1) // num_steps
    # 每个样本的第一个字符在corpus_indices中的下标
    example_indices = [i * num_steps for i in range(num_examples)]
    random.shuffle(example_indices)

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        # 得到当前batch的各个样本的首字符的下标
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)


# 测试一下这个函数，我们输入从0到29的连续整数作为一个人工序列，设批量大小和时间步数
# 分别为2和6，打印随机采样每次读取的小批量样本的输入X和标签Y。
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 保留下来的序列的长度
    corpus_len = len(corpus_indices) // batch_size * batch_size
    # 仅保留前corpus_len个字符
    corpus_indices = corpus_indices[: corpus_len]
    indices = torch.tensor(corpus_indices, device=device)
    # resize成(batch_size, )
    indices = indices.view(batch_size, -1)
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# 同样的设置下，打印相邻采样每次读取的小批量样本的输入X和标签Y。
# 相邻的两个随机小批量在原始序列上的位置相毗邻。
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
