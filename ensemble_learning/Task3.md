# Task3

## 1. 算法原理
### CART回归树
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/cart1.png)<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/cart2.png)<br>
<br>

### 集成原理
XGBoost的集成思想就是加法模型的思想，如下图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/theory1.png)<br>
对于一个m个样本n个特征的数据集<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/theory2.png)<br>
加法模型的计算结果如下：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/theory3.png)<br>
在这里，![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/theory4.png)
是CART回归树的空间，其中q代表每棵树的结构，这些树可以把一个样本分配到相应的叶子上；T是这棵树中叶子的数目。和决策树不同，
每一颗回归树在每一个叶子节点上都会有一个连续的分数，用w_i去表示第i个叶子上的分数。<br>
给定一个样本，我们可以用这些树中的决策规则（q）去把该样本分类到相应的叶子上，然后加和这些数上所有对应叶子的分数得到最终的计算结果。<br>

参考资料：<br>
1. [陈天奇论文](https://arxiv.org/pdf/1603.02754v1.pdf)<br>
<br>


## 2. 损失函数
为了学习模型中的函数集合，我们最小化如下的正则化过的目标函数：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/theory5.png)<br>
在这里，l是一个可微的凸损失函数，用来度量预测值和真实值之间的差距。第二项ohm惩罚模型的复杂度。这个附加的正则项有助于平滑最终的学习权重以防止过拟合。
















