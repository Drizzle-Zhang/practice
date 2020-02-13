# 线性回归  Softmax与分类模型  多层感知机

## 1 线性回归

### 1.0 矢量计算的优势

```Python
import torch
import time

# init variable a, b as 1000 dimension vector
n = 1000
a = torch.ones(n)
b = torch.ones(n)

# define a timer class to record time
class Timer(object):
	"""Record multiple running times."""
	def __init__(self):
		self.times = []
		self.start()
		
	def start(self):
		# start the timer
		self.start_time = time.time()
		
	def stop(self):
		# stop the timer and record time into a list
		self.times.append(time.time() - self.start_time)
		return self.times[-1]
	
	def avg(self):
		# calculate the average
		return sum(self.times)/len(self.times)
	
	def sum(self):
		# return the sum of the times
		return sum(self.times)
	
# for circulation
timer = Timer()
c = torch.zeros(n)
for i in range(n):
	c[i] = a[i] + b[i]
print('%.5f sec' % timer.stop()) 

        self.start()
        
    def start(self):
        # start the timer
        self.start_time = time.time()
        
    def stop(self):
        # stop the timer and record time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
    
    def avg(self):
        # calculate the average
        return sum(self.times)/len(self.times)
    
    def sum(self):
        # return the sum of the times
        return sum(self.times)
    
# for circulation
timer = Timer()
c = torch.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
print('%.5f sec' % timer.stop())

# verctor calculation
timer.start()
d = a + b
print('%.5f sec' % timer.stop())
# 0.01120 sec

# verctor calculation
timer.start()
d = a + b
print('%.5f sec' % timer.stop())
# 0.00027 sec
```

### 1.1 线性回归模型从零开始的实现

```python
# linear regression model without pytorch
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# simulate dataset
# set input feature number
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

```

![Image Name](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/scatterplot.png)

```python
# linear regression model without pytorch
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# simulate dataset
# set input feature number
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 构造一个张量，加入偏差
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)

# read dataset
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # disrupt the order of data
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)
        
# init model parameters
w = 0.01*torch.randn(num_inputs, 1, dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# w和b变成有梯度的复合变量，使得参数后续可以通过梯度下降方法优化
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# define linear model
linreg = lambda x, w, b: torch.mm(x, w) + b
squared_loss = lambda y_hat, y: (y_hat - y.view(y_hat.size())) ** 2 / 2
# view 返回一个有相同数据但大小不同的tensor; view(-1, n), -1是说该维度未定，由其它维度确定

# optimization function: stochastic gradient descent
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
        
# training
# hyper-parameters init
batch_size = 10
lr = 0.03
num_epoches = 5

net = linreg
loss = squared_loss

# training repeats num_epoches times
# in each epoch, all the samples in dataset will be used once
for epoch in range(num_epoches):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    
# training results
print(w, true_w)
print(b, true_b)
```

```
# output
epoch 1, loss 0.059993
epoch 2, loss 0.000320
epoch 3, loss 0.000052
epoch 4, loss 0.000051
epoch 5, loss 0.000051
tensor([[ 2.0004],
        [-3.3996]], requires_grad=True) [2, -3.4]
tensor([4.1994], requires_grad=True) 4.2
```

### 1.2 线性回归模型使用pytorch的简洁实现

```python
# linear regression model with pytorch
import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(
    np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# read dataset
from torch.utils import data

batch_size = 10

# combine featues and labels of dataset
dataset = data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2)

# define model
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        
    def forward(self, x):
        y = self.linear(x)
        return y
        
net = LinearNet(num_inputs)

# ways to init a multilayer network
# method one
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))
        
# initialization
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# loss function
loss = nn.MSELoss()

# optimization function
from torch import optim
optimizer = optim.SGD(net.parameters(), lr=0.03)

# training
num_epoches = 3
for epoch in range(num_epoches):
    for x, y in data_iter:
        y_hat = net(x)
        l = loss(y_hat, y.view(-1, 1))
        # reset gradients
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch + 1, l.item()))

# result comparision
dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)
```

```
# output
1.3.0
epoch 1, loss: 0.000580
epoch 2, loss: 0.000094
epoch 3, loss: 0.000062
[2, -3.4] tensor([[ 2.0002, -3.3995]])
4.2 tensor([4.2003])
```

## 2 Softmax与分类模型

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
# 添加引用模块的地址
sys.path.append(
    "C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL")
import d2lzh_pytorch as d2l
```

### 2.1 获取Fashion-MNIST训练集和读取数据

> torchvision主要由以下几部分构成：
>
> 1. torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
> 2. torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
> 3. torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
> 4. torchvision.utils: 其他的一些有用的方法。

```python
# Fashion-MNIST数据集的下载与导入
path_dataset = \
    'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL\Task01'
mnist_train = torchvision.datasets.FashionMNIST(
    root=path_dataset, train=True,
    download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(
    root=path_dataset, train=False,
    download=True, transform=transforms.ToTensor())
```

> class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
> - root（string）– 数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
> - train（bool, 可选）– 如果设置为True，从training.pt创建数据集，否则从test.pt创建。
> - download（bool, 可选）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
> - transform（可被调用 , 可选）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：transforms.RandomCrop。
> - target_transform（可被调用 , 可选）– 一种函数或变换，输入目标，进行变换。

```python
# show result
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 我们可以通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# 如果不做变换输入的数据是图像，我们可以看一下图片的类型参数：
mnist_PIL = torchvision.datasets.FashionMNIST(
    root=path_dataset, train=True, download=True)
PIL_feature, label = mnist_PIL[0]
print(PIL_feature)


# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(idx)] for idx in labels]


# 以子图形式展示一组图片数据集
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
```

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/mnist_example.png)

```python
x = []
y = []
for i in range(10):
    sub_x, sub_y = mnist_train[i]
    x.append(sub_x)
    y.append(sub_y)
show_fashion_mnist(x, get_fashion_mnist_labels(y))

# read dataset
batch_size = 256
train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
```



### 2.2 softmax

softmax运算符（softmax operator）通过下式将输出值o变换成值为正且和为1的概率分布：

$$
 \hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3) 
$$

其中

$$
\hat{y}1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\quad \hat{y}2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\quad \hat{y}3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
$$
这个方法将连续的神经网络输出层输出值转换为了离散的分类值。

```python
# 对多维Tensor按维度操作
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征

# output
tensor([[5, 7, 9]])
tensor([[ 6],
        [15]])
tensor([5, 7, 9])
tensor([ 6, 15])

# 定义softmax操作
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))

# output
tensor([[0.2253, 0.1823, 0.1943, 0.2275, 0.1706],
        [0.1588, 0.2409, 0.2310, 0.1670, 0.2024]]) 
 tensor([1.0000, 1.0000])

```

softmax回归模型
$$
\begin{aligned} \boldsymbol{o}^{(i)} &= \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\ \boldsymbol{\hat{y}}^{(i)} &= \text{softmax}(\boldsymbol{o}^{(i)}). \end{aligned}
$$

```python
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
```

### 2.3 交叉熵损失函数

在分类问题中，对于样本$i$，我们构造向量$\boldsymbol{y}^{(i)}\in \mathbb{R}^{q}$ ，使其第$y^{(i)}$（样本$i$类别的离散数值）个元素为1，其余为0。这样我们的训练目标可以设为使预测概率分布$\boldsymbol{\hat y}^{(i)}$尽可能接近真实的标签概率分布$\boldsymbol{y}^{(i)}$。

此时，交叉熵（cross entropy）是一个常用的衡量方法：


$$
H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},
$$


其中带下标的$y_j^{(i)}$是向量$\boldsymbol y^{(i)}$中非0即1的元素，需要注意将它与样本$i$类别的离散数值，即不带下标的$y^{(i)}$区分。在上式中，我们知道向量$\boldsymbol y^{(i)}$中只有第$y^{(i)}$个元素$y^{(i)}{y^{(i)}}$为1，其余全为0，于是$H(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}) = -\log \hat y_{y^{(i)}}^{(i)}$。也就是说，交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。当然，遇到一个样本有多个标签时，例如图像里含有不止一个物体时，我们并不能做这一步简化。但即便对于这种情况，交叉熵同样只关心对图像中出现的物体类别的预测概率。

假设训练数据集的样本数为$n$，交叉熵损失函数定义为 
$$
\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),
$$

其中$\boldsymbol{\Theta}$代表模型参数。同样地，如果每个样本只有一个标签，那么交叉熵损失可以简写成$\ell(\boldsymbol{\Theta}) = -(1/n) \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$。从另一个角度来看，我们知道最小化$\ell(\boldsymbol{\Theta})$等价于最大化$\exp(-n\ell(\boldsymbol{\Theta}))=\prod_{i=1}^n \hat y_{y^{(i)}}^{(i)}$，即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

```python
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))

# output
tensor([[0.1000],
        [0.5000]])

# define loss function
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
```

> torch.gather(input, dim, index, out=None) → Tensor     Gathers values along an axis specified by dim.
>
>     Parameters: 
>     
>         input (Tensor) – The source tensor
>         dim (int) – The axis along which to index
>         index (LongTensor) – The indices of elements to gather
>         out (Tensor, optional) – Destination tensor
>     
>     Example:
>     
>     >>> t = torch.Tensor([[1,2],[3,4]])
>     >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
>      1  1
>      4  3
>     [torch.FloatTensor of size 2x2]

### 2.4 softmax从零开始的实现

```python
# 定义准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# 获取训练集数据和测试集数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size, root=path_dataset)

# 模型参数初始化
num_inputs = 784    # 28*28
num_outputs = 10

W = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 训练模型
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零(是否使用了优化函数)
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr)

# 模型预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
```

```
# output
epoch 1, loss 0.7873, train acc 0.748, test acc 0.790
epoch 2, loss 0.5717, train acc 0.812, test acc 0.810
epoch 3, loss 0.5255, train acc 0.826, test acc 0.819
epoch 4, loss 0.5017, train acc 0.831, test acc 0.825
epoch 5, loss 0.4853, train acc 0.837, test acc 0.827
```

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/mnist_result.png)

### 2.5 softmax的简洁实现

```python
# pytorch version
# define net model
num_inputs = 784
num_outputs = 10

# 定义网络模型
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


# net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    # 用于数据size的转换
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


from collections import OrderedDict

net = nn.Sequential(
    # FlattenLayer(),
    # LinearNet(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),  # 变换层
        ('linear', nn.Linear(num_inputs, num_outputs))])  # 线性层
    # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
)

# 初始化模型参数
nn.init.normal_(net.linear.weight, mean=0, std=0.01)
nn.init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.CrossEntropyLoss()
# 下面是他的函数原型
# class torch.nn.CrossEntropyLoss(weight=None, size_average=None,
# ignore_index=-100, reduce=None, reduction='mean')

# 定义优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# 下面是函数原型
# class torch.optim.SGD(params, lr=, momentum=0, dampening=0,
# weight_decay=0, nesterov=False)

# training
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              None, None, optimizer)
```

## 3. 多层感知机

### 3.1 基本原理

#### 隐藏层

下图展示了一个多层感知机的神经网络图，它含有一个隐藏层，该层中有5个隐藏单元。

![Image Name](https://cdn.kesci.com/upload/image/q5ho684jmh.png)

#### 表达公式

具体来说，给定一个小批量样本$\boldsymbol{X} \in \mathbb{R}^{n \times d}$，其批量大小为$n$，输入个数为$d$。假设多层感知机只有一个隐藏层，其中隐藏单元个数为$h$。记隐藏层的输出（也称为隐藏层变量或隐藏变量）为$\boldsymbol{H}$，有$\boldsymbol{H} \in \mathbb{R}^{n \times h}$。因为隐藏层和输出层均是全连接层，可以设隐藏层的权重参数和偏差参数分别为$\boldsymbol{W}_h \in \mathbb{R}^{d \times h}$和 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，输出层的权重和偏差参数分别为$\boldsymbol{W}_o \in \mathbb{R}^{h \times q}$和$\boldsymbol{b}_o \in \mathbb{R}^{1 \times q}$。

我们先来看一种含单隐藏层的多层感知机的设计。其输出$\boldsymbol{O} \in \mathbb{R}^{n \times q}$的计算为


$$
 \begin{aligned} \boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\ \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o, \end{aligned}
$$


也就是将隐藏层的输出直接作为输出层的输入。如果将以上两个式子联立起来，可以得到


$$
 \boldsymbol{O} = (\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h)\boldsymbol{W}_o + \boldsymbol{b}_o = \boldsymbol{X} \boldsymbol{W}_h\boldsymbol{W}_o + \boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o. 
$$


从联立后的式子可以看出，虽然神经网络引入了隐藏层，却依然等价于一个单层神经网络：其中输出层权重参数为$\boldsymbol{W}_h\boldsymbol{W}_o$，偏差参数为$\boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o$。不难发现，即便再添加更多的隐藏层，以上设计依然只能与仅含输出层的单层神经网络等价。

#### 激活函数

上述问题的根源在于全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function）。

下面我们介绍几个常用的激活函数：

##### ReLU函数

ReLU（rectified linear unit）函数提供了一个很简单的非线性变换。给定元素$x$，该函数定义为


$$
\text{ReLU}(x) = \max(x, 0).
$$


可以看出，ReLU函数只保留正数元素，并将负数元素清零。为了直观地观察这一非线性变换，我们先定义一个绘图函数xyplot。

```python
# display Relu
def xyplot(x_vals, y_vals, name):
    # d2l.set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
```

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/relu.png)

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/relu_grad.png)

##### Sigmoid函数

```python
# sigmoid
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')
```

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/sigmoid.png)

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/s_grad.png)

##### tanh函数

```python
# tanh
y = x.tanh()
xyplot(x, y, 'tanh')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
```

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/tanh.png)

![](https://github.com/Drizzle-Zhang/practice/blob/master/deep_learning/Dive_into_DL/Task01/tanh_grad.png)

#### 关于激活函数的选择

ReLu函数是一个通用的激活函数，目前在大多数情况下适用。但是，ReLU函数只能在隐藏层中适用。

用于分类器时，sigmoid函数及其组合通常效果更好。由于梯度消失问题，有时要避免使用sigmoid和tanh函数。  

在神经网络层数较多的时候，最好使用ReLu函数，ReLu函数比较简单计算量少，而sigmoid和tanh函数计算量大很多。

在选择激活函数的时候可以先选用ReLu函数如果效果不理想可以尝试其他激活函数。

> sigmoid的梯度消失是指输入值特别大或者特别小的时候求出来的梯度特别小，当网络较深，反向传播时梯度一乘就没有了，这是sigmoid函数的饱和特性导致的。ReLU在一定程度上优化了这个问题是因为用了max函数，对大于0的输入直接给1的梯度，对小于0的输入则不管。
>
> 但是ReLU存在将神经元杀死的可能性，这和他输入小于0那部分梯度为0有关，当学习率特别大，对于有的输入在参数更新时可能会让某些神经元直接失活，以后遇到什么样的输入输出都是0，Leaky ReLU输入小于0的部分用很小的斜率，有助于缓解这个问题。

#### 多层感知机

多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。以单隐藏层为例并沿用本节之前定义的符号，多层感知机按以下方式计算输出：


$$
 \begin{aligned} \boldsymbol{H} &= \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\ \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o, \end{aligned} 
$$


其中$\phi$表示激活函数。

### 3.2 多层感知机从零开始的实现

> torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
> torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
>
> mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是(n×m)(n×m) 
> bmm是两个三维张量相乘, 两个输入tensor维度是(b×n×m)(b×n×m) , 第一维b代表batch size，输出为(b×n×p)(b×n×p)
> matmul可以进行张量乘法, 输入可以是高维.

```python
# 获取训练集数据和测试集数据
batch_size = 256
path_dataset = \
    'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL\Task01'
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size, root=path_dataset)

# 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# 定义网络
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# training
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

```
# output
epoch 1, loss 0.0031, train acc 0.710, test acc 0.758
epoch 2, loss 0.0019, train acc 0.820, test acc 0.839
epoch 3, loss 0.0016, train acc 0.846, test acc 0.852
epoch 4, loss 0.0015, train acc 0.856, test acc 0.848
epoch 5, loss 0.0014, train acc 0.864, test acc 0.832
```

#### 3.3 多层感知机pytorch实现

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = torch.nn.Sequential(
    d2l.FlattenLayer(),
    torch.nn.Linear(num_inputs, num_hiddens),
    torch.nn.ReLU(),
    torch.nn.Linear(num_hiddens, num_outputs),
)

for params in net.parameters():
    torch.nn.init.normal_(params, mean=0, std=0.01)

# training
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,
                                                    root=path_dataset)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, optimizer)
```

```
# output
epoch 1, loss 0.0030, train acc 0.709, test acc 0.795
epoch 2, loss 0.0019, train acc 0.818, test acc 0.823
epoch 3, loss 0.0017, train acc 0.844, test acc 0.790
epoch 4, loss 0.0015, train acc 0.857, test acc 0.843
epoch 5, loss 0.0014, train acc 0.866, test acc 0.829
```

