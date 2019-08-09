# GBDT算法梳理

## 1. 前向分布算法
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/FSA1.png)<br>


**Reference:**<br>
1. 李航，统计学习方法<br>
<br>


## 2. 负梯度拟合
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/futidunihe.png)<br>

## 3. 损失函数
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/loss_function.png)<br>

## 4. 回归
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/regression.png)<br>

## 5. 二分类，多分类
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/double_classification.png)<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/multi_classification.png)<br>

## 6. 正则化
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/normalization.png)<br>

## 7. 优缺点
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task2/advantages.png)<br>

## 8. sklearn参数
1. [sklearn梯度提升树(GBDT)调参小结](https://blog.csdn.net/lynn_001/article/details/85339034)<br>
2. [scikit-learn(sklearn)GBDT算法类库介绍](http://blog.sina.com.cn/s/blog_62970c250102xg5j.html)<br>
<br>

## 9. 应用场景
GBDT几乎可用于所有回归问题（线性/非线性），相对logistic regression仅能用于线性回归，GBDT的适用面非常广。亦可用于二分类问题（设定阈值，大于阈值为正例，反之为负例）。<br>
<br>
GBDT近些年因为被用于搜索引擎排序(RankNet)的机器学习模型而引起大家关注。<br>
搜索排序关注各个doc的顺序而不是绝对值，所以需要一个新的cost function，而RankNet基本就是在定义这个cost function，它可以兼容不同的算法（GBDT、神经网络...）。<br>
实际的搜索排序使用的是LambdaMART算法，必须指出的是由于这里要使用排序需要的cost function，LambdaMART迭代用的并不是残差。Lambda在这里充当替代残差的计算方法，它使用了一种类似Gradient x 步长模拟残差的方法。<br>
就像所有的机器学习一样，搜索排序的学习也需要训练集，这里一般是用人工标注实现，即对每一个(query,doc) pair给定一个分值（如1,2,3,4）,分值越高表示越相关，越应该排到前面。然而这些绝对的分值本身意义不大，例如你很难说1分和2分文档的相关程度差异是1分和3分文档差距的一半。相关度本身就是一个很主观的评判，标注人员无法做到这种定量标注，这种标准也无法制定。但标注人员很容易做到的是”AB都不错，但文档A比文档B更相关，所以A是4分，B是3分“。RankNet就是基于此制定了一个学习误差衡量方法，即cost function。具体而言，RankNet对任意两个文档A,B，通过它们的人工标注分差，用sigmoid函数估计两者顺序和逆序的概率P1。然后同理用机器学习到的分差计算概率P2（sigmoid的好处在于它允许机器学习得到的分值是任意实数值，只要它们的分差和标准分的分差一致，P2就趋近于P1）。这时利用P1和P2求的两者的交叉熵，该交叉熵就是cost function。它越低说明机器学得的当前排序越趋近于标注排序。为了体现NDCG的作用（NDCG是搜索排序业界最常用的评判标准），RankNet还在cost function中乘以了NDCG。<br>
好，现在我们有了cost function，而且它是和各个文档的当前分值yi相关的，那么虽然我们不知道它的全局最优方向，但可以求导求Gradient，Gradient即每个文档得分的一个下降方向组成的N维向量，N为文档个数（应该说是query-doc pair个数）。这里仅仅是把”求残差“的逻辑替换为”求梯度“，可以这样想：梯度方向为每一步最优方向，累加的步数多了，总能走到局部最优点，若该点恰好为全局最优点，那和用残差的效果是一样的。这时套到之前讲的逻辑，GDBT就已经可以上了。那么最终排序怎么产生呢？很简单，每个样本通过Shrinkage累加都会得到一个最终得分，直接按分数从大到小排序就可以了（因为机器学习产生的是实数域的预测分，极少会出现在人工标注中常见的两文档分数相等的情况，几乎不同考虑同分文档的排序方式）<br>
另外，如果feature个数太多，每一棵回归树都要耗费大量时间，这时每个分支时可以随机抽一部分feature来遍历求最优.<br>

**Reference:**<br>
1. [GBDT 入门教程之原理、所解决的问题、应用场景讲解](https://blog.csdn.net/molu_chase/article/details/78111148)<br>


