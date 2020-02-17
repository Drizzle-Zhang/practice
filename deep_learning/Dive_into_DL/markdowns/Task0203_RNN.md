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

