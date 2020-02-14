# 文本预处理  语言模型  循环神经网络

## 1 文本预处理

### 1.1 读入文本

```python
import collections
import re

with open('/home/kesci/input/timemachine7163/timemachine.txt', 'r') as f:
    # 对每句话，大写字母转为小写，a-z之外的符号全部替换为空格
    lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
print('# sentences %d' % len(lines))
print(lines[:10])
```

> re.sub(pattern, repl, string, count=0, flags=0)
>
> pattern：表示正则表达式中的模式字符串；
>
> repl：被替换的字符串（既可以是字符串，也可以是函数）；
>
> string：要被处理的，要被替换的字符串；
>
> count：匹配的次数, 默认是全部替换
>
> flags：

```
# output
# sentences 3221
['the time machine by h g wells ', '', '', '', '', 'i', '', '', 'the time traveller for so it will be convenient to speak of him ', 'was expounding a recondite matter to us his grey eyes shone and']
```

### 1.2 分词

```python
# 分词
tokens = [sentence.split(' ') for sentence in lines]
print(tokens[0:2])

# output
# [['the', 'time', 'machine', 'by', 'h', 'g', 'wells', ''], ['']]
```

### 1.3 建立字典

为了方便模型处理，我们需要将字符串转换为数字。因此我们需要先构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号。

```python
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
```

```
# output1
[('', 0), ('the', 1), ('time', 2), ('machine', 3), ('by', 4), ('h', 5), ('g', 6), ('wells', 7), ('i', 8), ('traveller', 9)]
4580

# output2
words: ['luxurious', 'after', 'dinner', 'atmosphere', 'when', 'thought', 'roams', 'gracefully']
indices: [65, 66, 67, 68, 69, 70, 71, 72]
words: ['free', 'of', 'the', 'trammels', 'of', 'precision', 'and', 'he', 'put', 'it', 'to', 'us', 'in', 'this']
indices: [73, 18, 1, 74, 18, 75, 30, 76, 77, 12, 16, 25, 44, 78]
```

### 1.4 分词工具

我们前面介绍的分词方式非常简单，它至少有以下几个缺点:

1. 标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
2. 类似“shouldn't", "doesn't"这样的词会被错误地处理
3. 类似"Mr.", "Dr."这样的词会被错误地处理

我们可以通过引入更复杂的规则来解决这些问题，但是事实上，有一些现有的工具可以很好地进行分词，我们在这里简单介绍其中的两个：[spaCy](https://spacy.io/)和[NLTK](https://www.nltk.org/)。

下面是一个简单的例子：

```python
text = "Mr. Chen doesn't agree with my suggestion."

# spaCy
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print([token.text for token in doc])

# output
['Mr.', 'Chen', 'does', "n't", 'agree', 'with', 'my', 'suggestion', '.']

# NLTK
from nltk.tokenize import word_tokenize
from nltk import data
data.path.append('/home/kesci/input/nltk_data3784/nltk_data')
print(word_tokenize(text))

# output
['Mr.', 'Chen', 'does', "n't", 'agree', 'with', 'my', 'suggestion', '.']
```

## 2. 语言模型

一段自然语言文本可以看作是一个离散时间序列，给定一个长度为$T$的词的序列$w_1, w_2, \ldots, w_T$，语言模型的目标就是评估该序列是否合理，即计算该序列的概率：


$$
P(w_1, w_2, \ldots, w_T).
$$
假设序列$w_1, w_2, \ldots, w_T$中的每个词是依次生成的，我们有


$$
\begin{align*}
P(w_1, w_2, \ldots, w_T)
&= \prod_{t=1}^T P(w_t \mid w_1, \ldots, w_{t-1})\\
&= P(w_1)P(w_2 \mid w_1) \cdots P(w_T \mid w_1w_2\cdots w_{T-1})
\end{align*}
$$


例如，一段含有4个词的文本序列的概率


$$
P(w_1, w_2, w_3, w_4) =  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3).
$$


语言模型的参数就是词的概率以及给定前几个词情况下的条件概率。设训练数据集为一个大型文本语料库，如维基百科的所有条目，词的概率可以通过该词在训练数据集中的相对词频来计算，例如，$w_1$的概率可以计算为：


$$
\hat P(w_1) = \frac{n(w_1)}{n}
$$


其中$n(w_1)$为语料库中以$w_1$作为第一个词的文本的数量，$n$为语料库中文本的总数量。

类似的，给定$w_1$情况下，$w_2$的条件概率可以计算为：


$$
\hat P(w_2 \mid w_1) = \frac{n(w_1, w_2)}{n(w_1)}
$$

其中$n(w_1, w_2)$为语料库中以$w_1$作为第一个词，$w_2$作为第二个词的文本的数量。

### 2.1 n元语法

序列长度增加，计算和存储多个词共同出现的概率的复杂度会呈指数级增加。$n$元语法通过马尔可夫假设简化模型，马尔科夫假设是指一个词的出现只与前面$n$个词相关，即$n$阶马尔可夫链（Markov chain of order $n$），如果$n=1$，那么有$P(w_3 \mid w_1, w_2) = P(w_3 \mid w_2)$。基于$n-1$阶马尔可夫链，我们可以将语言模型改写为


$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T P(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1}) .
$$


以上也叫$n$元语法（$n$-grams），它是基于$n - 1$阶马尔可夫链的概率语言模型。例如，当$n=2$时，含有4个词的文本序列的概率就可以改写为：


$$
\begin{align*}
P(w_1, w_2, w_3, w_4)
&= P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3)\\
&= P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) P(w_4 \mid w_3)
\end{align*}
$$


当$n$分别为1、2和3时，我们将其分别称作一元语法（unigram）、二元语法（bigram）和三元语法（trigram）。例如，长度为4的序列$w_1, w_2, w_3, w_4$在一元语法、二元语法和三元语法中的概率分别为


$$
\begin{aligned}
P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2) P(w_3) P(w_4) ,\\
P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) P(w_4 \mid w_3) ,\\
P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_2, w_3) .
\end{aligned}
$$

当$n$较小时，$n$元语法往往并不准确。例如，在一元语法中，由三个词组成的句子“你走先”和“你先走”的概率是一样的。然而，当$n$较大时，$n$元语法需要计算并存储大量的词频和多词相邻频率。

$n$元语法的缺陷：

1. 参数空间过大
2. 数据稀疏

### 2.2 语言模型数据集

```python
# read dataset
path_text = 'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\\' + \
            'Dive_into_DL\materials\Task02\jaychou_lyrics.txt'
with open(path_text, 'r', encoding='utf-8') as f:
    corpus_chars = f.read()
print(len(corpus_chars))
print(corpus_chars[: 40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[: 10000]
```

```
63282
想要有直升机
想要和你飞到宇宙去
想要和你融化在一起
融化在宇宙里
我每天每天每
```

```python
# build char2idx
# 去重，得到索引到字符的映射
idx_to_char = list(set(corpus_chars))
# 字符到索引的映射
char_to_idx = {char: i for i, char in enumerate(idx_to_char)}
print(char_to_idx[:5])
vocab_size = len(char_to_idx)
print(vocab_size)
# 将每个字符转化为索引，得到一个索引的序列
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[: 20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

```
1027
chars: 想要有直升机 想要和你飞到宇宙去 想要和
indices: [559, 490, 549, 248, 725, 977, 945, 559, 490, 97, 574, 937, 662, 794, 400, 603, 945, 559, 490, 97]
```

定义函数`load_data_jay_lyrics`，在后续章节中直接调用。

```python
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
```

### 2.3 时序数据的采样

在训练中我们需要每次随机读取小批量样本和标签。与之前章节的实验数据不同的是，时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”，即$X$=“想要有直升”，$Y$=“要有直升机”。

现在我们考虑序列“想要有直升机，想要和你飞到宇宙去”，如果时间步数为5，有以下可能的样本和标签：
* $X$：“想要有直升”，$Y$：“要有直升机”
* $X$：“要有直升机”，$Y$：“有直升机，”
* $X$：“有直升机，”，$Y$：“直升机，想”
* ...
* $X$：“要和你飞到”，$Y$：“和你飞到宇”
* $X$：“和你飞到宇”，$Y$：“你飞到宇宙”
* $X$：“你飞到宇宙”，$Y$：“飞到宇宙去”

可以看到，如果序列的长度为$T$，时间步数为$n$，那么一共有$T-n$个合法的样本，但是这些样本有大量的重合，我们通常采用更加高效的采样方式。我们有两种方式对时序数据进行采样，分别是随机采样和相邻采样。

#### 随机采样

下面的代码每次从数据里随机采样一个小批量。其中批量大小`batch_size`是每个小批量的样本数，`num_steps`是每个样本所包含的时间步数。
在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。

```python
import torch
import random


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符，python是0-base的
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
```

```
X:  tensor([[ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]]) 

X:  tensor([[ 0,  1,  2,  3,  4,  5],
        [18, 19, 20, 21, 22, 23]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [19, 20, 21, 22, 23, 24]]) 
```

#### 相邻采样

在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。

```python
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

```

```
X:  tensor([[ 0,  1,  2,  3,  4,  5],
        [15, 16, 17, 18, 19, 20]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [16, 17, 18, 19, 20, 21]]) 

X:  tensor([[ 6,  7,  8,  9, 10, 11],
        [21, 22, 23, 24, 25, 26]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],
        [22, 23, 24, 25, 26, 27]]) 
```

