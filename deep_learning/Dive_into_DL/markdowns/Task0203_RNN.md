# 循环神经网络

本节介绍循环神经网络，下图展示了如何基于循环神经网络实现语言模型。我们的目的是基于当前的输入与过去的输入序列，预测序列的下一个字符。循环神经网络引入一个隐藏变量$H$，用$H_{t}$表示$H$在时间步$t$的值。$H_{t}$的计算基于$X_{t}$和$H_{t-1}$，可以认为$H_{t}$记录了到当前字符为止的序列信息，利用$H_{t}$对序列的下一个字符进行预测。
![Image Name](https://cdn.kesci.com/upload/image/q5jkm0v44i.png?imageView2/0/w/640/h/640)

## 1 循环神经网络的构造

我们先看循环神经网络的具体构造。假设$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$是时间步$t$的小批量输入，$\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$是该时间步的隐藏变量，则：


$$
\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h).
$$


其中，$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$，$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$，$\boldsymbol{b}_{h} \in \mathbb{R}^{1 \times h}$，$\phi$函数是非线性激活函数。由于引入了$\boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$，$H_{t}$能够捕捉截至当前时间步的序列的历史信息，就像是神经网络当前时间步的状态或记忆一样。由于$H_{t}$的计算基于$H_{t-1}$，上式的计算是循环的，使用循环计算的网络即循环神经网络（recurrent neural network）。

在时间步$t$，输出层的输出为：


$$
\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q.
$$


其中$\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$，$\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$。

## 2 从零开始实现循环神经网络

我们先尝试从零开始实现一个基于字符级循环神经网络的语言模型，这里我们使用周杰伦的歌词作为语料，首先我们读入数据：

```python
import torch
import torch.nn as nn
import time
import math
import sys
sys.path.append(
    "C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_jay_lyrics():
    path_text = \
        'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\\' + \
        'Dive_into_DL\materials\Task02\jaychou_lyrics.txt'
    with open(path_text, 'r', encoding='utf-8') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(x.shape[0], n_class, dtype=dtype,
                         device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = \
    load_data_jay_lyrics()
x = torch.tensor([0, 2])
x_one_hot = one_hot(x, vocab_size)
print(x_one_hot)
print(x_one_hot.shape)
print(x_one_hot.sum(axis=1))
```

```
tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.]])
torch.Size([2, 1027])
tensor([1., 1.])
```

> #### one-hot向量
>
> 我们需要将字符表示成向量，这里采用one-hot向量。假设词典大小是$N$，每次字符对应一个从$0$到$N-1$的唯一的索引，则该字符的向量是一个长度为$N$的向量，若字符的索引是$i$，则该向量的第$i$个位置为$1$，其他位置为$0$。下面分别展示了索引为0和2的one-hot向量，向量长度等于词典大小。

> [PyTorch笔记之 scatter() 函数](https://www.cnblogs.com/dogecheng/p/11938009.html)

我们每次采样的小批量的形状是（批量大小, 时间步数）。下面的函数将这样的小批量变换成数个形状为（批量大小, 词典大小）的矩阵，矩阵个数等于时间步数。也就是说，时间步$t$的输入为$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$，其中$n$为批量大小，$d$为词向量大小，即one-hot向量长度（词典大小）。

```python
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)

```

```
5 torch.Size([2, 1027])
```

#### 初始化模型参数

```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# num_inputs: d
# num_hiddens: h, 隐藏单元的个数是超参数
# num_outputs: q


def get_params():
    def _one(shape):
        param = torch.zeros(shape, device=device, dtype=torch.float32)
        nn.init.normal_(param, 0, 0.01)
        return torch.nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))
    return (W_xh, W_hh, b_h, W_hq, b_q)
```

> w = torch.nn.Parameter(torch.FloatTensor(hidden_size)),首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个w 变成了模型的一部分，成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化

#### 定义模型

函数`rnn`用循环的方式依次完成循环神经网络每个时间步的计算。

```python
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


# 函数init_rnn_state初始化隐藏变量，这里的返回值是一个元组。
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 做个简单的测试来观察输出结果的个数（时间步数），
# 以及第一个时间步的输出层输出的形状和隐藏状态的形状。
print(X.shape)
print(num_hiddens)
print(vocab_size)
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(inputs), inputs[0].shape)
print(len(outputs), outputs[0].shape)
print(len(state), state[0].shape)
print(len(state_new), state_new[0].shape)
```

```
torch.Size([2, 5])
256
1027
5 torch.Size([2, 1027])
5 torch.Size([2, 1027])
1 torch.Size([2, 256])
1 torch.Size([2, 256])
```

> #### 裁剪梯度
>
> 循环神经网络中较容易出现梯度衰减或梯度爆炸(具体概念见有关章节)，这会导致网络几乎无法训练。裁剪梯度（clip gradient）是一种**应对梯度爆炸**的方法。假设我们把所有模型参数的梯度拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后的梯度
>
>
> $$
> \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}
> $$
>
>
> 的$L_2$范数不超过$\theta$。

```python
# 梯度裁剪
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
```

#### 定义预测函数

以下函数基于前缀`prefix`（含有数个字符的字符串）来预测接下来的`num_chars`个字符。这个函数稍显复杂，其中我们将循环神经单元`rnn`设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。

```python
# define prediction function
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    # output记录prefix加上预测的num_chars个字符
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])


# 我们先测试一下predict_rnn函数。我们将根据前缀“分开”创作长度为10个字符
# （不考虑前缀长度）的一段歌词。因为模型参数为随机值，所以预测结果也是随机的。
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            device, idx_to_char, char_to_idx)

```

```
'分开代爬领橱烧队江榜爽纵'
```

> #### 困惑度
>
> 我们通常使用困惑度（perplexity）来评价语言模型的好坏。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，
>
> * 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
> * 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
> * 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。
>
> 显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小`vocab_size`。
>
> 困惑度（perplexity）的基本思想是：**给测试集的句子赋予较高概率值的语言模型较好,当语言模型训练完之后，测试集中的句子都是正常的句子，那么训练好的模型就是在测试集上的概率越高越好，**公式如下：
> $$
> PP\left ( W \right )= \sqrt[N]{\frac{1}{P\left ( \omega _{1}\omega _{2} \cdot \cdot \cdot  \omega _{N} \right )}}
> $$
> 由公式可知，**句子概率越大，语言模型越好，迷惑度越小。**

### 定义模型训练函数

跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：

1. 使用困惑度评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。

```python
# training function
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # 如使用相邻采样，在epoch开始时初始化隐藏状态
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        # l_sum: sum of loss function
        # n: num of samples
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成形状为
            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应
            y = torch.flatten(Y.T)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params,
                                        init_rnn_state,
                                        num_hiddens, vocab_size, device,
                                        idx_to_char, char_to_idx))

```

> 随机采样时：每次迭代都需要重新初始化隐藏状态（每个epoch有很多词迭代，每次迭代都需要进行初始化，因为对于随机采样的样本中只有一个批量内的数据是连续的）
>
> 相邻采样时：如果是相邻采样，则说明前后两个batch的数据是连续的，所以在训练每个batch的时候只需要更新一次（也就是说模型在一个epoch中的迭代不需要重新初始化隐藏状态）
>
> detach了隐藏状态H。采用相邻采样的时候，当前这个batch的H来自上一个batch，如果没有在这个batch开始的时候把H（也就是H_{0}）从计算图当中分离出来，H的梯度的计算依赖于上一个batch的序列，而这一个过程会一直传递到最初的batch，所以随着batch的增加，计算H梯度的时间会越来越长。在batch开始的时候把H从计算图当中分离了，那就相当于是把上一个batch结束时的H的值作为当前batch的H的初始值，这个时候H是一个叶子，最后这个batch结束的时候，对H的梯度只会算到当前这个batch的序列起始处。
>
> 一般用grad.zero_grad()，创建grad.data变量，存储导数值
> 而grad.data.zero_()，如果没有设置requires_grad=True，grad.data变量是不存在的

#### 训练模型并创作歌词

现在我们可以训练模型了。首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。

```python
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# random sample
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

```

```
epoch 50, perplexity 69.294698, time 0.43 sec
 - 分开 一颗两 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一
 - 不分开 快颗用 一怪我 一子四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一颗四 一
epoch 100, perplexity 10.047098, time 0.43 sec
 - 分开 一只用双截棍 一直会我 说你说外 在一定梦 你一了纵 你一定空 你一了纵 你一定空 你一定纵 你一
 - 不分开堡 我有你和 你我已外乡对 一场看兮 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快使用双截棍
epoch 150, perplexity 2.887469, time 0.43 sec
 - 分开 一只都不三步四步 连成线背著背默默许在心愿 看远方的星如否听的见 手牵手 一步两步三步四步 连成线
 - 不分开吗 我爱你烦 你来我妈 这样了我的证据 情晶激的泪滴 闪烁成回忆 除人的美丽  没有你对不有 有只想
epoch 200, perplexity 1.569032, time 0.44 sec
 - 分开 装默心蜘教棍七百 她念让午险点阳B 瞎教堂午险边阳光射进教堂的角度 能知道你前世是狼人还起 你却形
 - 不分开吗 我叫你爸 你打我妈 这样对吗干嘛这样 还必让酒牵鼻子走 瞎 说笑我习多太就我 一和你 别怪的 丽
epoch 250, perplexity 1.296450, time 0.43 sec
 - 分开 让我爱好过 这样的美丽果 心所妙我的手友幽我 泪散的爹娘早已苍老了轮廓 娘子我欠你太多 一壶好酒
 - 不分开期把的胖女巫 用拉丁文念咒语啦啦呜 她养的黑猫笑起来像哭 啦啦啦呜 世吹来性发飘白 牵着你的手 一阵
```

```python
# consecutive sample
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

```
epoch 50, perplexity 57.514421, time 0.40 sec
 - 分开 我想要的爱写在西 我想你这样的让我 别有我有你你的一 悲想我这样的可爱 我不要再想 我不 我不要这
 - 不分开 我想要的可写女人 别想我有你的让我 我有你的可写 我有你的生写在西 我想你这样写着我的 有要我的爱
epoch 100, perplexity 6.670274, time 0.45 sec
 - 分开 我想要 爱你眼睛看着我 别发抖 一步两颗三颗四颗望著天 看星星 一颗两颗三颗四颗 连成线背著背默默
 - 不分开步 你已经很 我想多带 你不的梦 在有没梦 你一定梦 在一定梦 说一定梦 说一定梦 说一定梦 说一定
epoch 150, perplexity 2.018637, time 0.40 sec
 - 分开 一候我 你是我 手满球的 说 一定汉 一颗箱颗的留 它我得很你 伤故事 告诉我 印地安的传说 还真
 - 不分开觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生
epoch 200, perplexity 1.277430, time 0.40 sec
 - 分开 一候堂 说为我的脚有人亏 隔铁是声了写垂甜朽的可篇 我给你的爱写在西元前 深埋在美索不达米亚平原
 - 不分开觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生
epoch 250, perplexity 1.166600, time 0.41 sec
 - 分开 一候了 是诉于依旧代日折黑瓦的淡淡的忧伤 消失的 旧时光 一九四三 回头看 的片段 有一些风霜 老
 - 不分开觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好生活 我该好好生
```



## 3 循环神经网络的简介实现

### 定义模型

我们使用Pytorch中的`nn.RNN`来构造循环神经网络。在本节中，我们主要关注`nn.RNN`的以下几个构造函数参数：

* `input_size` - The number of expected features in the input x
* `hidden_size` – The number of features in the hidden state h
* `nonlinearity` – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
* `batch_first` – If True, then the input and output tensors are provided as (batch_size, num_steps, input_size). Default: False

这里的`batch_first`决定了输入的形状，我们使用默认的参数`False`，对应的输入形状是 (num_steps, batch_size, input_size)。

`forward`函数的参数为：

* `input` of shape (num_steps, batch_size, input_size): tensor containing the features of the input sequence. 
* `h_0` of shape (num_layers * num_directions, batch_size, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.

`forward`函数的返回值是：

* `output` of shape (num_steps, batch_size, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
* `h_n` of shape (num_layers * num_directions, batch_size, hidden_size): tensor containing the hidden state for t = num_steps.

现在我们构造一个`nn.RNN`实例，并用一个简单的例子来看一下输出的形状。

```python
# RNN with pytorch
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
num_steps, batch_size = 35, 2
X = torch.rand(num_steps, batch_size, vocab_size)
state = None
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)
```

```
torch.Size([35, 2, 256]) torch.Size([1, 2, 256])
```

```python
# 定义一个完整的基于循环神经网络的语言模型
# define model
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state
```

```python
# prediction
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])


# 使用权重为随机值的模型来预测一次
model = RNNModel(rnn_layer, vocab_size).to(device)
predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)

```

```
'分开替视吃蛦旁苦苦苦鼠蜘'
```

```python
# train; consecutive sample
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size,
                                              num_steps, device)  # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state[0].detach_()
                    state[1].detach_()
                else:
                    state.detach_()
            (output, state) = model(X,
                                    state)  # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.T)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))
                

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)

```

```
epoch 50, perplexity 11.145100, time 0.25 sec
 - 分开 一场悲剧 我想要的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可
 - 不分开 我想你的手 我的 我想你 不过我不 我不了我不 我不多再想 我不要再想 我不要再想 我不 我不多再
epoch 100, perplexity 1.283560, time 0.25 sec
 - 分开的风美 不知道 就是开不了口模著一定实我的难空 却只想你的可爱女人 温柔的让我心疼的可爱女人 透明的
 - 不分开不了你说 是是我听你 黑色的没有你 我说了飞 一九四三 泛头看抽现 还有什么满天 干什么 干什么 东
epoch 150, perplexity 1.066973, time 0.22 sec
 - 分开 我轻的的溪边 我都笔 太多 我 别反方 那么面阵风难怎么  我 从以后跟你的让我感动的可爱女人 坏
 - 不分开不了你说 是不听听你 会怪我 你过我不多口  为什么我想我想要你却  我有些你 我笑要再想要 直
epoch 200, perplexity 1.052460, time 0.22 sec
 - 分开 我轻的的家边 后知道 我后再会多 几天 只是上怕的黑白美主随的可爱 我想要你想微为每妈妈到 什么都
 - 不分开 我是你的黑笑笑能承受我已你了可爱 我的温暖  快什么分妈出 干什么 干什么 我沉下 天都当有 在等
epoch 250, perplexity 1.020480, time 0.22 sec
 - 分开 我不要气相 黑色幽默 说通 你又再考倒我 说散 你想很久了吧? 败给你的黑色幽默 不想太多 我想一
 - 不分开 我不开不悲剧 印可以让我满上为一九四三 泛黄的春联还残己养前 那么上到几小 我不要再说散 我想这故
```

## 4 LSTM的理论理解

### 从RNN说起

循环神经网络（Recurrent Neural Network，RNN）是一种用于处理序列数据的神经网络。相比一般的神经网络来说，他能够处理序列变化的数据。比如某个单词的意思会因为上文提到的内容不同而有不同的含义，RNN就能够很好地解决这类问题。

### 普通RNN

先简单介绍一下一般的RNN。  

其主要形式如下图所示（图片均来自台大李宏毅教授的PPT）：

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/RNN.jpg)

这里：  

![[公式]](https://www.zhihu.com/equation?tex=x) 为当前状态下数据的输入， ![[公式]](https://www.zhihu.com/equation?tex=h)  表示接收到的上一个节点的输入。  

![[公式]](https://www.zhihu.com/equation?tex=y) 为当前节点状态下的输出，而  ![[公式]](https://www.zhihu.com/equation?tex=h%27) 为传递到下一个节点的输出。  

通过上图的公式可以看到，输出 **h'** 与 **x** 和 **h** 的值都相关。  

而 **y** 则常常使用 **h'** 投入到一个线性层（主要是进行维度映射）然后使用softmax进行分类得到需要的数据。  

对这里的**y**如何通过 **h'** 计算得到往往看具体模型的使用方式。  

通过序列形式的输入，我们能够得到如下形式的RNN。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/RNN2.jpg)

### LSTM结构梳理

长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。  

LSTM结构（图右）和普通RNN的主要输入输出区别如下所示。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/LSTM1.jpg)

相比RNN只有一个传递状态  ![[公式]](https://www.zhihu.com/equation?tex=h%5Et+) ，LSTM有两个传输状态，一个  ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) （cell state），和一个  ![[公式]](https://www.zhihu.com/equation?tex=h%5Et) （hidden state）。（Tips：RNN中的 ![[公式]](https://www.zhihu.com/equation?tex=h%5Et) 对于LSTM中的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) ）  

其中对于传递下去的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) 改变得很慢，通常输出的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) 是上一个状态传过来的 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D) 加上一些数值。  

而 ![[公式]](https://www.zhihu.com/equation?tex=h%5Et) 则在不同节点下往往会有很大的区别。

下面具体对LSTM的内部结构来进行剖析。 

首先使用LSTM的当前输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) 和上一个状态传递下来的 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D) 拼接训练得到四个状态。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/LSTM2.jpg)

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/LSTM3.jpg)

其中， ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef+) ， ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) ，![[公式]](https://www.zhihu.com/equation?tex=z%5Eo) 是由拼接向量乘以权重矩阵之后，再通过一个 ![[公式]](https://www.zhihu.com/equation?tex=sigmoid+) 激活函数转换成0到1之间的数值，来作为一种门控状态。而  ![[公式]](https://www.zhihu.com/equation?tex=z)  则是将结果通过一个 ![[公式]](https://www.zhihu.com/equation?tex=tanh) 激活函数将转换成-1到1之间的值（这里使用 ![[公式]](https://www.zhihu.com/equation?tex=tanh) 是因为这里是将其做为输入数据，而不是门控信号）。  

**下面开始进一步介绍这四个状态在LSTM内部的使用。**

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/LSTM4.jpg)

![[公式]](https://www.zhihu.com/equation?tex=%5Codot) 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。 ![[公式]](https://www.zhihu.com/equation?tex=%5Coplus) 则代表进行矩阵加法。

LSTM内部主要有三个阶段： 

1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行**选择性**忘记。简单来说就是会 “忘记不重要的，记住重要的”。  具体来说是通过计算得到的 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef) （f表示forget）来作为忘记门控，来控制上一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D) 哪些需要留哪些需要忘。  

2. 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。主要是会对输入  ![[公式]](https://www.zhihu.com/equation?tex=x%5Et)  进行选择记忆。哪些重要则着重记录下来，哪些不重要，则少记一些。当前的输入内容由前面计算得到的  ![[公式]](https://www.zhihu.com/equation?tex=z+)  表示。而选择的门控信号则是由 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) （i代表information）来进行控制。将上面两步得到的结果相加，即可得到传输给下一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Et) 。也就是上图中的第一个公式。   

3. 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。主要是通过  ![[公式]](https://www.zhihu.com/equation?tex=z%5Eo)  来进行控制的。并且还对上一阶段得到的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Eo) 进行了放缩（通过一个tanh激活函数进行变化）。  

与普通RNN类似，输出 ![[公式]](https://www.zhihu.com/equation?tex=y%5Et) 往往最终也是通过 ![[公式]](https://www.zhihu.com/equation?tex=h%5Et) 变化得到。 

参考：[人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)

## 5 GRU的理论理解

GRU（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。

相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。

### GRU的输入输出结构

GRU的输入输出结构与普通的RNN是一样的。

有一个当前的输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) ，和上一个节点传递下来的隐状态（hidden state） ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D) ，这个隐状态包含了之前节点的相关信息。

结合 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et+) 和 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D)，GRU会得到当前隐藏节点的输出 ![[公式]](https://www.zhihu.com/equation?tex=y%5Et+) 和传递给下一个节点的隐状态 ![[公式]](https://www.zhihu.com/equation?tex=h%5Et) 。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/GRU1.jpg)

### GRU的内部结构

首先，我们先通过上一个传输下来的状态 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D) 和当前节点的输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) 来获取两个门控状态。如下图2-2所示，其中 ![[公式]](https://www.zhihu.com/equation?tex=r+) 控制重置的门控（reset gate）， ![[公式]](https://www.zhihu.com/equation?tex=z) 为控制更新的门控（update gate）。

> Tips： ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) 为*[sigmoid](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Sigmoid_function)*函数，通过这个函数可以将数据变换为0-1范围内的数值，从而来充当门控信号。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/GRU2.jpg)

**与LSTM分明的层次结构不同，下面将对GRU进行一气呵成的介绍~~~ 请大家屏住呼吸，不要眨眼。**

得到门控信号之后，首先使用重置门控来得到**“重置”**之后的数据 ![[公式]](https://www.zhihu.com/equation?tex=%7Bh%5E%7Bt-1%7D%7D%27+%3D+h%5E%7Bt-1%7D+%5Codot+r+) ，再将 ![[公式]](https://www.zhihu.com/equation?tex=%7Bh%5E%7Bt-1%7D%7D%27) 与输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et+) 进行拼接，再通过一个[tanh](https://link.zhihu.com/?target=https%3A//baike.baidu.com/item/tanh)激活函数来将数据放缩到**-1~1**的范围内。即得到如下图2-3所示的 ![[公式]](https://www.zhihu.com/equation?tex=h%27) 。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/GRU3.jpg)

这里的 ![[公式]](https://www.zhihu.com/equation?tex=h%27+) 主要是包含了当前输入的 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) 数据。有针对性地对 ![[公式]](https://www.zhihu.com/equation?tex=h%27) 添加到当前的隐藏状态，相当于”记忆了当前时刻的状态“。类似于LSTM的选择记忆阶段（参照我的上一篇文章）。

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/materials/Task03/GRU4.jpg)

![[公式]](https://www.zhihu.com/equation?tex=%5Codot) 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。 ![[公式]](https://www.zhihu.com/equation?tex=%5Coplus) 则代表进行矩阵加法操作。

最后介绍GRU最关键的一个步骤，我们可以称之为**”更新记忆“**阶段。

在这个阶段，我们同时进行了遗忘和记忆两个步骤。我们使用了先前得到的更新门控 ![[公式]](https://www.zhihu.com/equation?tex=z) （update gate）。

**更新表达式**： ![[公式]](https://www.zhihu.com/equation?tex=h%5Et+%3D+z+%5Codot+h%5E%7Bt-1%7D+%2B+%281+-+z%29%5Codot+h%27) 

首先再次强调一下，门控信号（这里的 ![[公式]](https://www.zhihu.com/equation?tex=z) ）的范围为0~1。门控信号越接近1，代表”记忆“下来的数据越多；而越接近0则代表”遗忘“的越多。



GRU很聪明的一点就在于，**我们使用了同一个门控 ![[公式]](https://www.zhihu.com/equation?tex=z) 就同时可以进行遗忘和选择记忆（LSTM则要使用多个门控）**。

- ![[公式]](https://www.zhihu.com/equation?tex=z+%5Codot+h%5E%7Bt-1%7D) ：表示对原本隐藏状态的选择性“遗忘”。这里的 ![[公式]](https://www.zhihu.com/equation?tex=z) 可以想象成遗忘门（forget gate），忘记 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D) 维度中一些不重要的信息。
- ![[公式]](https://www.zhihu.com/equation?tex=%281-z%29+%5Codot+h%27) ： 表示对包含当前节点信息的 ![[公式]](https://www.zhihu.com/equation?tex=h%27) 进行选择性”记忆“。与上面类似，这里的 ![[公式]](https://www.zhihu.com/equation?tex=%281-z%29) 同理会忘记 ![[公式]](https://www.zhihu.com/equation?tex=h+%27) 维度中的一些不重要的信息。或者，这里我们更应当看做是对 ![[公式]](https://www.zhihu.com/equation?tex=h%27+) 维度中的某些信息进行选择。
- ![[公式]](https://www.zhihu.com/equation?tex=h%5Et+%3D+z+%5Codot+h%5E%7Bt-1%7D+%2B+%281+-+z%29%5Codot+h%27) ：结合上述，这一步的操作就是忘记传递下来的 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D+) 中的某些维度信息，并加入当前节点输入的某些维度信息。

可以看到，这里的遗忘 ![[公式]](https://www.zhihu.com/equation?tex=z) 和选择 ![[公式]](https://www.zhihu.com/equation?tex=%281-z%29) 是联动的。也就是说，对于传递进来的维度信息，我们会进行选择性遗忘，则遗忘了多少权重 （![[公式]](https://www.zhihu.com/equation?tex=z) ），我们就会使用包含当前输入的 ![[公式]](https://www.zhihu.com/equation?tex=h%27) 中所对应的权重进行弥补 ![[公式]](https://www.zhihu.com/equation?tex=%281-z%29) 。以保持一种”恒定“状态。

### LSTM与GRU的关系

大家看到 ![[公式]](https://www.zhihu.com/equation?tex=r) (reset gate)实际上与他的名字有点不符。我们仅仅使用它来获得了 ![[公式]](https://www.zhihu.com/equation?tex=h%E2%80%99) 。

那么这里的 ![[公式]](https://www.zhihu.com/equation?tex=h%27) 实际上可以看成对应于LSTM中的hidden state；上一个节点传下来的 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bt-1%7D) 则对应于LSTM中的cell state。z对应的则是LSTM中的 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ef) forget gate，那么 ![[公式]](https://www.zhihu.com/equation?tex=%281-z%29) 我们似乎就可以看成是选择门 ![[公式]](https://www.zhihu.com/equation?tex=z%5Ei) 了。

参考：[人人都能看懂的GRU](https://zhuanlan.zhihu.com/p/32481747)

## 6 GRU的实现

RNN存在的问题：梯度较容易出现衰减或爆炸（BPTT）  
⻔控循环神经⽹络：捕捉时间序列中时间步距离较⼤的依赖关系  
**RNN**:  


![Image Name](https://cdn.kesci.com/upload/image/q5jjvcykud.png?imageView2/0/w/320/h/320)


$$
H_{t} = ϕ(X_{t}W_{xh} + H_{t-1}W_{hh} + b_{h})
$$
**GRU**:


![Image Name](https://cdn.kesci.com/upload/image/q5jk0q9suq.png?imageView2/0/w/640/h/640)



$$
R_{t} = σ(X_tW_{xr} + H_{t−1}W_{hr} + b_r)\\    
Z_{t} = σ(X_tW_{xz} + H_{t−1}W_{hz} + b_z)\\  
\widetilde{H}_t = tanh(X_tW_{xh} + (R_t ⊙H_{t−1})W_{hh} + b_h)\\
H_t = Z_t⊙H_{t−1} + (1−Z_t)⊙\widetilde{H}_t
$$
• 重置⻔有助于捕捉时间序列⾥短期的依赖关系；  
• 更新⻔有助于捕捉时间序列⾥⻓期的依赖关系。    

### 载入数据集

```python
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("../input/")
import d2l_jay9460 as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_jay_lyrics():
    path_text = '/home/zy/my_git/practice/deep_learning/Dive_into_DL/' \
                'materials/Task02/jaychou_lyrics.txt'
    with open(path_text, 'r', encoding='utf-8') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
```

### 初始化参数

```python
# 初始化参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device,
                          dtype=torch.float32)  # 正态分布
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device,
                                               dtype=torch.float32),
                                   requires_grad=True))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(
        torch.zeros(num_outputs, device=device, dtype=torch.float32),
        requires_grad=True)
    return nn.ParameterList(
        [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def init_gru_state(batch_size, num_hiddens, device):  # 隐藏状态初始化
    return (torch.zeros((batch_size, num_hiddens), device=device),)

```

### GRU模型

```python
# GRU模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + R * torch.matmul(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

### 训练模型

```python
# 训练模型
num_epochs, num_steps, batch_size, lr, clipping_theta = \
    160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)

```

```
epoch 40, perplexity 154.950742, time 0.91 sec
 - 分开 我想你的让我的爱爱人 我想你的让我不 我想你的让我不想想想想想你想你的爱爱人 我想你的让我不 我想
 - 不分开 我想你的让我不 我想你的让我不想想想想想你想你的爱爱人 我想你的让我不 我想你的让我不想想想想想你
epoch 80, perplexity 33.035831, time 0.71 sec
 - 分开 一直在人截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快
 - 不分开 爱你在我不多 让你在我 别你 这样的节笑 你说 却想我 别你的手 我不要再想 我不要再想 我不要再
epoch 120, perplexity 4.981806, time 0.85 sec
 - 分开我 一定球 快沉我抬起 一直走 停给我抬开头 有话去对医药箱说 别怪我 别怪我 说你怎么面对我 甩开
 - 不分开  我来你烦 我有多烦恼  没有你在我有多难熬多恼  没有你烦 我有多烦恼  没有你在我有多难熬多恼
epoch 160, perplexity 1.465822, time 0.86 sec
 - 分开 一个中酒 你的它美主义 还生水起 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快使用双截棍 哼
 - 不分开 你已经离 我不多烦恼  没有你烦我有多烦恼多难熬  穿过云层 我试著努力向你奔跑 爱才送到 你却已
```

### 简洁实现

```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

```
epoch 40, perplexity 1.020338, time 0.46 sec
 - 分开始想像 爸和妈当年的模样 说著一口吴侬软语的姑娘缓缓走过外滩 消失的 旧时光 一九四三 回头看 的片
 - 不分开始 担心今天的你过得好不好 整个画面是你 想你想的睡不著 嘴嘟嘟那可爱的模样 还有在你身上香香的味道
epoch 80, perplexity 1.013508, time 0.45 sec
 - 分开始想要 我的快乐是你 想你想的都会笑 没有你在 我有多难熬  没有你在我有多难熬多烦恼  没有你烦
 - 不分开不可以简简单单没有伤害 你 靠着我的肩膀 你 在我胸口睡著 像这样的生活 我爱你 你爱我 我想大声宣
epoch 120, perplexity 1.008346, time 0.49 sec
 - 分开始想像 爸和妈当年的模样 说著一口吴侬软语的姑娘缓缓走过外滩 消失的 旧时光 一九四三 在回忆 的路
 - 不分开始 担心今天的你过得好不好 整个画面是你 想你想的睡不著 嘴嘟嘟那可爱的模样 还有在你身上香香的味道
epoch 160, perplexity 1.008036, time 0.45 sec
 - 分开始的话 你甘会听 不要再这样打我妈妈 难道你手不会痛吗 其实我回家就想要阻止一切 让家庭回到过去甜甜
 - 不分开始 担心今天的你过得好不好 整个画面是你 想你想的睡不著 嘴嘟嘟那可爱的模样 还有在你身上香香的味道
```

## 7 LSTM的实现

长短期记忆long short-term memory :  
遗忘门:控制上一时间步的记忆细胞 
输入门:控制当前时间步的输入  
输出门:控制从记忆细胞到隐藏状态  
记忆细胞：⼀种特殊的隐藏状态的信息的流动  


![Image Name](https://cdn.kesci.com/upload/image/q5jk2bnnej.png?imageView2/0/w/640/h/640)

$$
I_t = σ(X_tW_{xi} + H_{t−1}W_{hi} + b_i) \\
F_t = σ(X_tW_{xf} + H_{t−1}W_{hf} + b_f)\\
O_t = σ(X_tW_{xo} + H_{t−1}W_{ho} + b_o)\\
\widetilde{C}_t = tanh(X_tW_{xc} + H_{t−1}W_{hc} + b_c)\\
C_t = F_t ⊙C_{t−1} + I_t ⊙\widetilde{C}_t\\
H_t = O_t⊙tanh(C_t)
$$

### 初始化参数

```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device,
                          dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device,
                                               dtype=torch.float32),
                                   requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(
        torch.zeros(num_outputs, device=device, dtype=torch.float32),
        requires_grad=True)
    return nn.ParameterList(
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
         W_hq, b_q])


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

```

### LSTM模型

```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)
```

### 训练模型

```python
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
```

```
epoch 40, perplexity 211.108955, time 1.13 sec
 - 分开 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我
 - 不分开 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我 我不的我
epoch 80, perplexity 63.426335, time 1.14 sec
 - 分开 我想你的你 我不要这我 我不要我 我不要我 我不不觉 我不不觉 我不不觉 我不不觉 我不不觉 我不
 - 不分开 我想你的你 我不要这我 我不要我 我不要我 我不不觉 我不不觉 我不不觉 我不不觉 我不不觉 我不
epoch 120, perplexity 15.437350, time 1.16 sec
 - 分开 你说你的太笑 我 却你你的你笑 像这样 说你的睛笑 就你 你想很久了吧? 我想你 你给我 说你 是
 - 不分开 我想你你已经 有你 在不样的太笑 我想想你想想 我想 你不 我不 我不 我不 我不要 爱你的对快快
epoch 160, perplexity 4.124116, time 1.05 sec
 - 分开 我已带你 我有一定婆 一起看起 你知了这我 我知好好生活 一静悄觉 又过了一个秋 后知后觉 我该好
 - 不分开 你已经你 我跟好好熬我 我该好觉生活 静静悄觉默离离到 在入了危默怪到  却去了了我不要难熬烦我
```

### 简洁实现

```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

```
epoch 40, perplexity 1.021080, time 0.52 sec
 - 分开始移动 一阵莫名感动 我想带你看着我的肩膀 你 在我胸口睡著 像这样的生活 我爱你 你爱我 我想大声
 - 不分开始打呼啸而过 是谁说没有 有一条热昏头的响尾蛇 无力的躺在干枯的河 在等待雨季来临变沼泽 灰狼啃食著
epoch 80, perplexity 1.016574, time 0.52 sec
 - 分开始移动 一阵莫名感动 我想带你看着我的手不放开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想
 - 不分开始打呼雨  是一场悲剧 我可以让生命就这样毫无意义 或许在最后能听到你一句 轻轻的叹息  后悔着对不
epoch 120, perplexity 1.008828, time 0.52 sec
 - 分开始打呼 管家是一只会说法语举止优雅的猪 吸血前会念约翰福音做为弥补 拥有一双蓝色眼睛的凯萨琳公主 专
 - 不分开始打呼啸管家是一是那么 你想就这样牵着你的手不放开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我
epoch 160, perplexity 1.010719, time 0.52 sec
 - 分开始玩笑 想通 却又再考倒我 说散 你想很久了吧? 败给你的黑色幽默 说散 你想很久了吧? 我的认真败
 - 不分开 爱能不能够永远单纯没有悲哀 我 想带你骑单车 我 想和你看棒球 想这样没担忧 唱着歌 一直走 我想
```

## 8 深度循环神经网络

![Image Name](https://cdn.kesci.com/upload/image/q5jk3z1hvz.png?imageView2/0/w/320/h/320)


$$
\boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)} + \boldsymbol{b}_h^{(1)})\\
\boldsymbol{H}_t^{(\ell)} = \phi(\boldsymbol{H}_t^{(\ell-1)} \boldsymbol{W}_{xh}^{(\ell)} + \boldsymbol{H}_{t-1}^{(\ell)} \boldsymbol{W}_{hh}^{(\ell)} + \boldsymbol{b}_h^{(\ell)})\\
\boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hq} + \boldsymbol{b}_q
$$
深度循环神经网络的层数并不是越多越好

```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

```
epoch 40, perplexity 48.529632, time 0.94 sec
 - 分开你 我不要再我的人不人的美两步三颗步四步的语截鸠 可爱女人 可爱女人 可爱女人 可爱女人 可爱女人
 - 不分开的让我疯狂 我不要我 一子在美四棍 我想你 我不要我 一子在美四棍 我想你 我不要我 一子在美四棍
epoch 80, perplexity 1.620074, time 0.96 sec
 - 分开有了这样的甜蜜 让我开始乡相信命运 感谢地心引力 让我碰到你 漂亮的让我面红的可爱女人 温柔的让我心
 - 不分开有了口让我感动的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女人 坏坏的让我疯狂的可爱女
epoch 120, perplexity 1.064946, time 0.93 sec
 - 分开有多难过 说你回方的可爱女人 温柔的让我心疼的可爱女人 透明的让我感动的可爱女人 坏坏的让我疯狂的可
 - 不分开有多难过 说你眼  我不懂不说你说 想要你的微笑每天都能看到  我知道这里很美但家乡的你更美走过了很
epoch 160, perplexity 1.023880, time 0.96 sec
 - 分开有多难 我不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 又过了一个秋 后知后觉 我该好好
 - 不分开有多难  一直在掉落在 身找 还你连一九四三 泛黄的春联还残待雨季来临变沼泽 灰狼啃食著水鹿的骨头
```

## 2 双向循环神经网络

双向循环神经网络是利用上下文的信息去构建语言模型

![Image Name](https://cdn.kesci.com/upload/image/q5j8hmgyrz.png?imageView2/0/w/320/h/320)

$$
\begin{aligned} \overrightarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(f)} + \overrightarrow{\boldsymbol{H}}_{t-1} \boldsymbol{W}_{hh}^{(f)} + \boldsymbol{b}_h^{(f)})\\
\overleftarrow{\boldsymbol{H}}_t &= \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(b)} + \overleftarrow{\boldsymbol{H}}_{t+1} \boldsymbol{W}_{hh}^{(b)} + \boldsymbol{b}_h^{(b)}) \end{aligned}
$$
$$
\boldsymbol{H}t=(\overrightarrow{\boldsymbol{H}}{t}, \overleftarrow{\boldsymbol{H}}_t)
$$
$$
\boldsymbol{O}t = \boldsymbol{H}t \boldsymbol{W}{hq} + \boldsymbol{b}q
$$
```python
num_hiddens=128
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,bidirectional=True)
model = d2l.RNNModel(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)
```

```
epoch 40, perplexity 1.001762, time 0.66 sec
 - 分开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开
 - 不分开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开
epoch 80, perplexity 1.000562, time 0.60 sec
 - 分开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开
 - 不分开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开
epoch 120, perplexity 1.000281, time 0.60 sec
 - 分开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开
 - 不分开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开
epoch 160, perplexity 1.000168, time 0.60 sec
 - 分开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开始开
 - 不分开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开不开
```

说明双向神经网络结果不一定好
