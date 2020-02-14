#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: Text_process.py
# @time: 2020/2/13 10:37

import collections
import re

# 读入文本
with open('/home/kesci/input/timemachine7163/timemachine.txt', 'r') as f:
    # 对每句话，大写字母转为小写，a-z之外的符号全部替换为空格
    lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
print('# sentences %d' % len(lines))
print(lines[:10])


# 分词
tokens = [sentence.split(' ') for sentence in lines]
print(tokens[0:2])


def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    # 返回一个字典，记录每个词的出现次数
    return collections.Counter(tokens)


# vocabulary
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # :
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            # padding 用于补短句使得输入成为矩阵
            # bos eos在句子开头与结尾增加的特殊token
            # unk 语料库里没有的词
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk = 0
            self.idx_to_token += ['unk']
        self.idx_to_token += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # word to index
        if not isinstance(tokens, (list, tuple)):
            # tokens是个字符串，在token_to_idx里找tokens，如找不到，返回self.unk
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        # index to word
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
print(len(vocab))

# 将词转化为索引
for i in range(15, 17):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
