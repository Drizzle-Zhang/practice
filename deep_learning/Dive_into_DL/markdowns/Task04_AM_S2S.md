# 注意力机制

在“编码器—解码器（seq2seq）”⼀节⾥，解码器在各个时间步依赖相同的背景变量（context vector）来获取输⼊序列信息。当编码器为循环神经⽹络时，背景变量来⾃它最终时间步的隐藏状态。将源序列输入信息以循环单位状态编码，然后将其传递给解码器以生成目标序列。然而这种结构存在着问题，尤其是**RNN机制实际中存在长程梯度消失的问题**，对于较长的句子，我们很难寄希望于将输入的序列转化为定长的向量而保存所有的有效信息，所以随着所需翻译句子的长度的增加，这种结构的效果会显著下降。

与此同时，解码的目标词语可能只与原输入的部分词语有关，而并不是与所有的输入有关。例如，当把“Hello world”翻译成“Bonjour le monde”时，“Hello”映射成“Bonjour”，“world”映射成“monde”。在seq2seq模型中，解码器只能隐式地从编码器的最终状态中选择相应的信息。然而，注意力机制可以将这种选择过程显式地建模。

![Image Name](https://cdn.kesci.com/upload/image/q5km4dwgf9.PNG?imageView2/0/w/960/h/960)

## 1 注意力机制框架

Attention 是一种通用的带权池化方法，输入由两部分构成：询问（query）和键值对（key-value pairs）。$𝐤_𝑖∈ℝ^{𝑑_𝑘}, 𝐯_𝑖∈ℝ^{𝑑_𝑣}$. Query  $𝐪∈ℝ^{𝑑_𝑞}$ , attention layer得到输出与value的维度一致 $𝐨∈ℝ^{𝑑_𝑣}$. 对于一个query来说，attention layer 会与每一个key计算注意力分数并进行权重的归一化，输出的向量$o$则是value的加权求和，而每个key计算的权重与value一一对应。

为了计算输出，我们首先假设有一个函数$\alpha$ 用于计算query和key的相似性，然后可以计算所有的 attention scores $a_1, \ldots, a_n$ by


$$
a_i = \alpha(\mathbf q, \mathbf k_i).
$$


我们使用 softmax函数 获得注意力权重：


$$
b_1, \ldots, b_n = \textrm{softmax}(a_1, \ldots, a_n).
$$


最终的输出就是value的加权求和：


$$
\mathbf o = \sum_{i=1}^n b_i \mathbf v_i.
$$


![Image Name](https://cdn.kesci.com/upload/image/q5km4ooyu2.PNG?imageView2/0/w/960/h/960)

不同的attetion layer的区别在于score函数的选择，在本节的其余部分，我们将讨论两个常用的注意层 Dot-product Attention 和 Multilayer Perceptron Attention；随后我们将实现一个引入attention的seq2seq模型并在英法翻译语料上进行训练与测试。

### Softmax屏蔽

在深入研究实现之前，我们首先介绍softmax操作符的一个屏蔽操作。

因为为了统一句子长度，之前加入了padding符号。但是对于AM，不需要考虑padding，所以将其变为负无穷

```python
import math
import torch
import torch.nn as nn


# Softmax屏蔽
def SequenceMask(X, X_len, value=-1e6):
    maxlen = X.size(1)
    # print(X.size(),torch.arange((maxlen),dtype=torch.float)[None, :],'\n',X_len[:, None] )
    mask = torch.arange((maxlen), dtype=torch.float)[None, :] >= X_len[:, None]
    # print(mask)
    X[mask] = value
    return X


def masked_softmax(X, valid_length):
    # X: 3-D tensor, batch_size*seq_len*dim
    # valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.FloatTensor(
                    valid_length.numpy().repeat(shape[1], axis=0))
                # [2,3] -> [2,2,3,3] 对于最大长度为2的batch来说
                # 作用是和变换形状后的X维数一致
            except:
                valid_length = torch.FloatTensor(
                    valid_length.cpu().numpy().repeat(shape[1],
                                                      axis=0))  # [2,2,3,3]
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length)

        return softmax(X).reshape(shape)


masked_softmax(torch.rand((2, 2, 4), dtype=torch.float),
               torch.FloatTensor([2, 3]))
```

```
tensor([[[0.4566, 0.5434, 0.0000, 0.0000],
         [0.4910, 0.5090, 0.0000, 0.0000]],

        [[0.2537, 0.3907, 0.3556, 0.0000],
         [0.3347, 0.2245, 0.4408, 0.0000]]])
```

**超出2维矩阵的乘法** 

$X$ 和 $Y$ 是维度分别为$(b,n,m)$ 和$(b, m, k)$的张量，进行 $b$ 次二维矩阵乘法后得到 $Z$, 维度为 $(b, n, k)$。


$$
Z[i,:,:] = dot(X[i,:,:], Y[i,:,:])\qquad for\ i= 1,…,n\ .
$$

> 高维张量的矩阵乘法可用于并行计算多个位置的注意力分数。

```python
torch.bmm(torch.ones((2,1,3), dtype = torch.float), torch.ones((2,3,2), dtype = torch.float))
```

```
tensor([[[3., 3.]],

        [[3., 3.]]])
```

## 点积注意力

The dot product 假设query和keys有相同的维度, 即 $\forall i, 𝐪,𝐤_𝑖 ∈ ℝ_𝑑 $. 通过计算query和key转置的乘积来计算attention score,通常还会除去 $\sqrt{d}$ 减少计算出来的score对维度𝑑的依赖性，如下


$$
𝛼(𝐪,𝐤)=⟨𝐪,𝐤⟩/ \sqrt{d} 
$$

假设 $ 𝐐∈ℝ^{𝑚×𝑑}$ 有 $m$ 个query，$𝐊∈ℝ^{𝑛×𝑑}$ 有 $n$ 个keys. 我们可以通过矩阵运算的方式计算所有 $mn$ 个score：


$$
𝛼(𝐐,𝐊)=𝐐𝐊^𝑇/\sqrt{d}
$$

现在让我们实现这个层，它支持一批查询和键值对。此外，它支持作为正则化随机删除一些注意力权重.

```python
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_length: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key

        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        print("attention_weight\n", attention_weights)
        return torch.bmm(attention_weights, value)

```

#### 测试

现在我们创建了两个批，每个批有一个query和10个key-values对。我们通过valid_length指定，对于第一批，我们只关注前2个键-值对，而对于第二批，我们将检查前6个键-值对。因此，尽管这两个批处理具有相同的查询和键值对，但我们获得的输出是不同的。

```python
# example
atten = DotProductAttention(dropout=0)

keys = torch.ones((2,10,2),dtype=torch.float)
values = torch.arange((40), dtype=torch.float).view(1,10,4).repeat(2,1,1)
atten(torch.ones((2,1,2),dtype=torch.float), keys, values, torch.FloatTensor([2, 6]))

```

```
tensor([[[0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000]],

        [[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000,
          0.0000, 0.0000]]])
tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],

        [[10.0000, 11.0000, 12.0000, 13.0000]]])
```

## 多层感知机注意力

在多层感知器中，我们首先将 query and keys 投影到  $ℝ^ℎ$ .为了更具体，我们将可以学习的参数做如下映射 
$𝐖_𝑘∈ℝ^{ℎ×𝑑_𝑘}$ ,  $𝐖_𝑞∈ℝ^{ℎ×𝑑_𝑞}$ , and  $𝐯∈ℝ^h$ . 将score函数定义
$$
𝛼(𝐤,𝐪)=𝐯^𝑇tanh(𝐖_𝑘𝐤+𝐖_𝑞𝐪)
$$

然后将key 和 value 在特征的维度上合并（concatenate），然后送至 a single hidden layer perceptron 这层中 hidden layer 为  ℎ  and 输出的size为 1 .隐层激活函数为tanh，无偏置.

```python
class MLPAttention(nn.Module):
    def __init__(self, units,ipt_dim,dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes.
        self.W_k = nn.Linear(ipt_dim, units, bias=False)
        self.W_q = nn.Linear(ipt_dim, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length):
        query, key = self.W_k(query), self.W_q(key)
        #print("size",query.size(),key.size())
        # expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast.
        features = query.unsqueeze(2) + key.unsqueeze(1)
        #print("features:",features.size())  #--------------开启
        scores = self.v(features).squeeze(-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return torch.bmm(attention_weights, value)

```

#### 测试

尽管MLPAttention包含一个额外的MLP模型，但如果给定相同的输入和相同的键，我们将获得与DotProductAttention相同的输出

```
atten = MLPAttention(ipt_dim=2, units=8, dropout=0)
atten(torch.ones((2, 1, 2), dtype=torch.float), keys, values,
      torch.FloatTensor([2, 6]))

```

```
tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],

        [[10.0000, 11.0000, 12.0000, 13.0000]]], grad_fn=<BmmBackward>)
```

> 在Dot-product Attention中，key与query维度需要一致，在MLP Attention中则不需要。

## 2 对注意力机制的理解

### Seq2Seq模型

我们知道Seq2Seq模型的结构是基于编码器-解码器，可以解决输入和输出序列不等长的问题，例如机器翻译问题。编码器和解码器本质上是两个RNN，其中编码器对输入序列进行分析编码成一个上下文向量(Context vector)，解码器利用这个编码器生成的向量根据具体任务来进行解码，得到一个新的序列。

#### 编码器

如下图所示为一个编码器的结构，就是将输入序列x1x1 x_1 至x4x4 x_4 依次输入到编码器中得到了h1h1 h_1 至h4h4 h_4 的隐含状态，而最终的上下文向量cc c ,可以是编码器最后一个时间步的隐藏状态，也可以是编码器每个时间步得到的隐藏状态进行一个函数映射(就是使用某个度量函数去表示原始序列的信息)，这个上下文向量后面会再解码器生成序列中。

![preview](https://pic2.zhimg.com/v2-03aaa7754bb9992858a05bb9668631a9_r.jpg) 

#### 解码器

下图是两种比较常见的Seq2Seq模型的结构，两个图的左半部分都是上面所说的编码器部分，而右半部分就是解码器部分了。如下面第一张图所示，其直接将编码器的输出作为解码器的初始隐藏状态，然后直接进行解码。第二张图是直接将编码器得到的上下文向量输入到解码器的每个时间步中，并且每个时间步的上下文向量是相同，换句话说就是解码器每个时间步都使用了相同的上下文向量。这两种情况可能带来的问题是，当需要编码的句子太长的时候，由于上下文向量能够存储信息的容量是有限的，所以可能会导致，信息的丢失，此外，解码器每个时间步的上下文向量都是一个相同的对输入序列的表征，对于上面两种问题，基于注意力机制的Seq2Seq模型给了很好的解决办法。

![preview](https://pic4.zhimg.com/v2-77e8a977fc3d43bec8b05633dc52ff9f_r.jpg) 

![preview](https://pic4.zhimg.com/v2-e0fbb46d897400a384873fc100c442db_r.jpg) 

### Attention机制的Seq2Seq

基于Attention的Seq2Seq模型本质上就是在上述的图三中的解码器部分进行了改进，在解码器的每个时间步上使用不同的上下文向量c 如下图所示的$c_1,c_2,c_3$ ，但是对于解码器的初始化一般还是会使用编码器最后时间步的隐藏状态，即图中的$h'_0=c $(此处的c表示的是编码器最后时间步的隐藏状态)，如何得到解码器不同时间步不同的上下文向量就是Attention要做的事情了。

![preview](https://pic2.zhimg.com/v2-8da16d429d33b0f2705e47af98e66579_r.jpg) 

Attention机制生成的上下文向量可以自动的去选取与当前时间步输出最有用的信息，用有限的上下文向量的容量去表示当前时间步对输入信息最关注的那部分信息，**最简单的做法就是对编码器输出的所有时间步的隐藏状态进行一个加权平均，不同的权值所对应的隐含状态就是对不同时间步的输入信息关的注程度**，下面的点积模型示意图可以形象的表示该过程。图中的a 表示是编码器不同时间步对应的权值，而其权值又决定于编码器该时间步的隐藏状态以及解码器上一个时间步的隐藏状态，因此注意力层在训练过程中的变化是由隐藏状态决定的，即决定隐藏状态的参数的优化也包含了注意力层的信息。下面给出一个简单的解释：设解码器当前隐藏状态为$s_{t'} $

则无注意力的解码器当前的隐藏状态表示为：$s_{t'} = g(y_{t'-1}, c, s_{t'-1}) $
基于注意力的解码器当前的隐藏状态表示为：$s_{t'} = g(y_{t'-1}, c_{t'}, s_{t'-1}) $

其中：

$y_{t′−1}$::  解码器上一时间步的输出
c:  编码器最后时间步(或者之前所有时间步隐藏状态的某种映射)的隐藏状态
$c_{t'}$:  解码器在t'时间步通过注意力机制获得的的上下文向量
$s_{t'-1}$: 解码器的解码器的 解码器的 $t'-1$时间步的隐藏状态

下面两个图是背景变量$c_t'$ 的生成过程，最后就剩下如何计算$a_{ij}$ 的值了。这里的a其实$a_{ij}$ 是注意力打分函数的输出，跟三部分东西有关系，分别是查询项q(quary)q(quary) q(quary) ：解码器上一时间步的隐藏状态 $s_{t'-1}$ ，键项k(key)和值项v(value)都是编码器的隐含状态$(h_1, h_2, h_3 )$。常见的注意力打分函数有：

`加性模型`：$s_{(x_i,q)}= v^Ttanh(Wx_i+Uq) $
`点积模型`： $s_{(x_i,q)}= x_i^Tq $
`双线性模型`：$s_{(x_i,q)}= x_i^TW$q$

![preview](https://pic.imgdb.cn/item/5e4c98c548b86553ee7bde57.jpg) 

`点积模型`可视化如下：

![preview](https://pic1.zhimg.com/v2-d266bf48a1d77e7e4db607978574c9fc_r.jpg) 

最后基于注意力的Seq2Seq模型可以用下图进行表示：

![Image Name](https://pic.imgdb.cn/item/5e4c991148b86553ee7be9be.png)

参考：

[完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)

[Seq2seq模型及注意力机制](http://www.ryluo.cn/2020/02/17/Seq2seq模型及注意力机制/)

[图解神经机器翻译中的注意力机制](https://zhuanlan.zhihu.com/p/56704058)

## 3 引入注意力机制的Seq2seq模型

本节中将注意机制添加到sequence to sequence 模型中，以显式地使用权重聚合states。下图展示encoding 和decoding的模型结构，在时间步为t的时候。此刻attention layer保存着encodering看到的所有信息——即encoding的每一步输出。在decoding阶段，解码器的$t$时刻的隐藏状态被当作query，encoder的每个时间步的hidden states作为key和value进行attention聚合. Attetion model的输出当作成上下文信息context vector，并与解码器输入$D_t$拼接起来一起送到解码器：

![Image Name](https://cdn.kesci.com/upload/image/q5km7o8z93.PNG?imageView2/0/w/800/h/800)

$$
Fig1具有注意机制的seq-to-seq模型解码的第二步
$$


下图展示了seq2seq机制的所以层的关系，下面展示了encoder和decoder的layer结构

![Image Name](https://cdn.kesci.com/upload/image/q5km8dihlr.PNG?imageView2/0/w/800/h/800)

$$
Fig2具有注意机制的seq-to-seq模型中的层结构
$$

### 解码器

   由于带有注意机制的seq2seq的编码器与之前章节中的Seq2SeqEncoder相同，所以在此处我们只关注解码器。我们添加了一个MLP注意层(MLPAttention)，它的隐藏大小与解码器中的LSTM层相同。然后我们通过从编码器传递三个参数来初始化解码器的状态:

- the encoder outputs of all timesteps：encoder输出的各个状态，被用于attetion layer的memory部分，有相同的key和values


- the hidden state of the encoder’s final timestep：编码器最后一个时间步的隐藏状态，被用于初始化decoder 的hidden state


- the encoder valid length: 编码器的有效长度，借此，注意层不会考虑编码器输出中的填充标记（Paddings）


   在解码的每个时间步，我们使用解码器的最后一个RNN层的输出作为注意层的query。然后，将注意力模型的输出与输入嵌入向量连接起来，输入到RNN层。虽然RNN层隐藏状态也包含来自解码器的历史信息，但是attention model的输出显式地选择了enc_valid_len以内的编码器输出，这样attention机制就会尽可能排除其他不相关的信息。

```python
class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = MLPAttention(num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers,
                           dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        #         print("first:",outputs.size(),hidden_state[0].size(),hidden_state[1].size())
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.permute(1, 0, -1), hidden_state, enc_valid_len)
        # outputs.swapaxes(0, 1)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        # ("X.size",X.size())
        X = self.embedding(X).transpose(0, 1)
        #         print("Xembeding.size2",X.size())
        outputs = []
        for l, x in enumerate(X):
            #             print(f"\n{l}-th token")
            #             print("x.first.size()",x.size())
            # query shape: (batch_size, 1, hidden_size)
            # select hidden state of the last rnn layer as query
            query = hidden_state[0][-1].unsqueeze(
                1)  # np.expand_dims(hidden_state[0][-1], axis=1)
            # context has same shape as query
            #             print("query enc_outputs, enc_outputs:\n",query.size(), enc_outputs.size(), enc_outputs.size())
            context = self.attention_cell(query, enc_outputs, enc_outputs,
                                          enc_valid_len)
            # Concatenate on the feature dimension
            #             print("context.size:",context.size())
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            # Reshape x to (1, batch_size, embed_size+hidden_size)
            #             print("rnn",x.size(), len(hidden_state))
            out, hidden_state = self.rnn(x.transpose(0, 1), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.transpose(0, 1), [enc_outputs, hidden_state,
                                         enc_valid_len]

```

现在我们可以用注意力模型来测试seq2seq。为了与第9.7节中的模型保持一致，我们对vocab_size、embed_size、num_hiddens和num_layers使用相同的超参数。结果，我们得到了相同的解码器输出形状，但是状态结构改变了。

```python
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers,
                           dropout=dropout)

    def begin_state(self, batch_size, device):
        return [
            torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),
                        device=device),
            torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),
                        device=device)]

    def forward(self, X, *args):
        X = self.embedding(X)  # X shape: (batch_size, seq_len, embed_size)
        X = X.transpose(0, 1)  # RNN needs first axes to be time
        # state = self.begin_state(X.shape[1], device=X.device)
        out, state = self.rnn(X)
        # The shape of out is (seq_len, batch_size, num_hiddens).
        # out: 每个rnn单元的输出；是一个序列
        # state contains the hidden state and the memory cell of the last
        # time step, the shape is (num_layers, batch_size, num_hiddens)
        return out, state


encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8,
                            num_hiddens=16, num_layers=2)
# encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8,
                                  num_hiddens=16, num_layers=2)
X = torch.zeros((4, 7),dtype=torch.long)
print("batch size=4\nseq_length=7\nhidden dim=16\nnum_layers=2\n")
print('encoder output size:', encoder(X)[0].size())
print('encoder hidden size:', encoder(X)[1][0].size())
print('encoder memory size:', encoder(X)[1][1].size())
state = decoder.init_state(encoder(X), None)
out, state = decoder(X, state)
out.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape

```

```
batch size=4
seq_length=7
hidden dim=16
num_layers=2

encoder output size: torch.Size([7, 4, 16])
encoder hidden size: torch.Size([2, 4, 16])
encoder memory size: torch.Size([2, 4, 16])

(torch.Size([4, 7, 10]), 3, torch.Size([4, 7, 16]), 2, torch.Size([2, 4, 16]))
```

### 训练与预测

从结果中我们可以看出，由于训练数据集中的序列相对较短，额外的注意层并没有带来显著的改进。由于编码器和解码器的注意层的计算开销，该模型比没有注意的seq2seq模型慢得多。

```python
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.0
batch_size, max_len = 64, 10
lr, num_epochs, ctx = 0.005, 500, d2l.try_gpu()
path_txt = "/home/zy/my_git/practice/deep_learning/Dive_into_DL/" \
           "materials/Task04/fraeng6506/fra.txt"

src_vocab, tgt_vocab, train_iter = \
    d2l.load_data_nmt(path_txt, batch_size, max_len, 50000)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.EncoderDecoder(encoder, decoder)


d2l.train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)


for sentence in ['Go .', 'Good Night !', "I'm OK .", 'I won !']:
    print(sentence + ' => ' + d2l.predict_s2s_ch9(
        model, sentence, src_vocab, tgt_vocab, max_len, ctx))

```

