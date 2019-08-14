# LightGBM算法梳理

## 1. 简介
LightGBM （Light Gradient Boosting Machine）是一个实现 GBDT 算法的框架，支持高效率的并行训练，并且具有以下优点：<br>
更快的训练速度 <br>
更低的内存消耗 <br>
更好的准确率 <br>
分布式支持，可以快速处理海量数据 <br><br>

## 2. LightGBM的起源
传统的boosting算法（如GBDT和XGBoost）已经有相当好的效率，但是在如今的大样本和高维度的环境下，传统的boosting似乎在效率和可扩展性上不能满足现在的需求了，主要的原因就是传统的boosting算法需要对每一个特征都要扫描所有的样本点来选择最好的切分点，这是非常的耗时。为了解决这种在大样本高纬度数据的环境下耗时的问题，出现了Lightgbm 。<br>
Lightgbm使用了如下两种解决办法：一是GOSS（Gradient-based One-Side Sampling, 基于梯度的单边采样），不是使用所用的样本点来计算梯度，而是对样本进行采样来计算梯度；二是EFB（Exclusive Feature Bundling， 互斥特征捆绑） ，这里不是使用所有的特征来进行扫描获得最佳的切分点，而是将某些特征进行捆绑在一起来降低特征的维度，是寻找最佳切分点的消耗减少。这样大大的降低的处理样本的时间复杂度，但在精度上，通过大量的实验证明，在某些数据集上使用Lightgbm并不损失精度，甚至有时还会提升精度。<br>
<br>

### 2.1 单边梯度采样算法（Grandient-based One-Side Sampling，GOSS）
LightGBM使用GOSS算法进行训练样本采样的优化。在AdaBoost算法中，采用了增加被错误分类的样本的权重来优化下一次迭代时对哪些样本进行重点训练。然而GBDT算法中没有样本的权重，但是LightGBM采用了基于每个样本的梯度进行训练样本的优化，具有较大梯度的数据对计算信息增益的贡献比较大。当一个样本点的梯度很小，说明该样本的训练误差很小，即该样本已经被充分训练。然而在计算过程中，仅仅保留梯度较大的样本（例如：预设置一个阈值，或者保留最高若干百分位的梯度样本），抛弃梯度较小样本，会改变样本的分布并且降低学习的精度。GOSS算法的提出很好的解决了这个问题。<br><br>
GOSS算法的基本思想是首先对训练集数据根据梯度排序，预设一个比例，保留在所有样本中梯度高于该比例的数据样本；梯度低于该比例的数据样本不会直接丢弃，而是设置一个采样比例，从梯度较小的样本中按比例抽取样本。为了弥补对样本分布造成的影响，GOSS算法在计算信息增益时，会对较小梯度的数据集乘以一个系数，用来放大。这样，在计算信息增益时，算法可以更加关注“未被充分训练”的样本数据。<br><br>
具体的算法如下：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/GOSS.jpg)<br>
回归树采用方差增益作为选取分裂特征的指标，通过GOSS算法对样本数据集进行采样后，生成梯度较大的数据样本子集A，梯度较小数据样本子集B，根据子集计算方差增益，此处GOSS通过对较小的样本数据集估算增益，大大的减少了计算量。而且通过证明，GOSS算法不会过多的降低训练的精度。<br>

### 2.2 Exclusive Feature Bundling 算法(EFB)
LightGBM算法不仅通过GOSS算法对训练样本进行采样优化，也进行了特征抽取，以进一步优化模型的训练速度。但是这里的特征抽取与特征提取还不一样，并不减少训练时数据特征向量的维度，而是将互斥特征绑定在一起，从而减少特征维度。该算法的主要思想是：假设通常高维度的数据往往也是稀疏的，而且在稀疏的特征空间中，大量的特征是互斥的，也就是，它们不会同时取到非0值。这样，可以安全的将互斥特征绑定在一起形成一个单一的特征包（称为Exclusive Feature Bundling）。这样，基于特征包构建特征直方图的复杂度由O(#data*#feature)变为O(#data*#bundle).<br><br>
* 怎么判断哪些特征应该被绑定在一起呢？<br>

LightGBM将这个问题转化成为了图着色问题：给定一个无向图，所有的特征视为图G的顶点集合V，如果两个特征之间不是互斥，使用一个边将它们连接，E代表所有不是互斥特征之间边的集合。使用贪心算法解决该着色问题，结果会将互斥的特征放入相同的集合内（相同的颜色），每个集合即为一个特征包。实际上，有很多特征，虽然不是100%的互斥，但是基本上不会同时取到非0值。所以在LightGBM的算法中，会允许特征有一部分的冲突，这样可以生成更小的特征包集合，进一步减少计算时的特征规模，提高效率。假设变量代表一个特征包内最大的冲突率。那么选用一个相对较小的值，算法可以在效率和精确度之间有更好的平衡。<br>
基于以上的思路，LightGBM设计了一个这样的算法（Greedy Bundling）：首先，使用训练集的特征构建一个加权图，图的边的权重值为两个特征间的整体冲突。然后，根据图节点（特征）的度对特征进行降序排序。最后，检查每一个排列好的特征，将其在较小的冲突率下（值来控制）分配到一个已存在的特征包内，或者新创建一个特征包（这里判断是否新创建一个特征包，是由算法中的参数K来决定的，K代表了特征包内最大的冲突数）。这样生成特征包的复杂度为。这种处理只在训练前进行一次。因此，该复杂度在feature规模不是很大的情况下，是可以接受的。但是当训练集的特征规模达到百万级或者以上时，就无法忍受了。因此，为了进一步提高算法的效率，LightGBM采用了更加高效的不用构建图的排序策略：按非零值计数排序，因为更多的非零的特征值会导致更高的冲突概率。具体的算法流程如下图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/EFB.jpg)<br>
<br>
* EFB算法如何绑定特征？<br>
上面解决了如何判断哪些特征要被绑定在一起，那么EFB算法如何绑定特征呢？如何既减少了特征维度，又保证原始的特征值可以在特征包中被识别出来呢？由于LightGBM是采用直方图算法减少对于寻找最佳分裂点的算法复杂度，直方图算法将特征值离散到若干个bin中。这里EFB算法为了保留特征，将bundle内不同的特征加上一个偏移常量，使不同特征的值分布到bundle的不同bin内。例如：特征A的取值范围为[0,10)，特征B的原始取值范围为[0，20)，对特征B的取值上加一个偏置常量10，将其取值范围变为[10,30)，这样就可以将特征A和B绑定在一起了。具体的算法流程如下图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/EFB1.jpg)<br>
EFB算法可以将数据集中互斥的特征绑定在一起，形成低维的特征集合，能够有效的避免对0值特征的计算。实际上，在算法中，可以对每个特征建立一个记录非零值特征的表格。通过对表中数据的扫描，可以有效的降低创建直方图的时间复杂度（从到)。在LightGBM的算法中也确实使用了这种优化方式，当Bundle稀疏时，这个优化与EFB并不冲突。<br><br>


参考资料：<br>
1. [机器学习算法梳理-LightGBM](https://blog.csdn.net/mingxiaod/article/details/86233309)<br>
2. [论文原文](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/LightGBM.pdf)<br>
2. [LightGBM算法初探](https://cloud.tencent.com/developer/news/375910)<br>
<br>

## 3. Histogram VS pre-sorted

### 3.1 Histogram 算法
直方图算法的基本思想是先把连续的浮点特征值离散化成k个整数，同时构造一个宽度为k的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。 <br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/histogram.png)<br>

### 3.2 pre-sorted算法
xgboost算法是基于预排序方法，这种构建决策树的算法基本思想是：<br>

* 首先，对所有特征都按照特征的数值进行预排序。<br>
* 其次，在遍历分割点的时候用O(data)的代价找到一个特征上的最好分割点。<br>
* 最后，找到最好的分割点后，将数据分裂成左右子节点。<br>

这样做能精确地找到分割点。但是空间消耗大，时间消耗也大，对cache优化不好。预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化。同时，在每一层，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样。<br>

### 3.2 Histogram与pre-sorted的比较
#### 优势
* Pre-sorted 算法需要的内存约是训练数据的两倍(2 * #data * #features* 4Bytes)，它需要用32位浮点(4Bytes)来保存 feature value，并且对每一列特征，都需要一个额外的排好序的索引，这也需要32位(4Bytes)的存储空间。因此是(2 * #data * #features* 4Bytes)。而对于 histogram 算法，则只需要(#data * #features * 1Bytes)的内存消耗，仅为 pre-sorted算法的1/8。因为 histogram 算法仅需要存储 feature bin value (离散化后的数值)，不需要原始的 feature value，也不用排序，而 bin value 用 1Bytes(256 bins) 的大小一般也就足够了。<br>
* 计算上的优势则是大幅减少了计算分割点增益的次数。对于每一个特征，pre-sorted 需要对每一个不同特征值都计算一次分割增益，代价是O(#feature*#distinct_values_of_the_feature)；而 histogram 只需要计算#bins次，代价是O(#feature*#bins)。<br>
* 还有一个很重要的点是cache-miss。事实上，cache-miss对速度的影响是特别大的。预排序中有2个操作频繁的地方会造成cache miss，一是对梯度的访问，在计算gain的时候需要利用梯度，不同特征访问梯度的顺序都是不一样的，且是随机的，因此这部分会造成严重的cache-miss。二是对于索引表的访问，预排序使用了一个行号到叶子节点号的索引表（row_idx_to_tree_node_idx ），来防止数据切分时对所有的数据进行切分，即只对该叶子节点上的样本切分。在与level-wise进行结合的时候， 每一个叶子节点都要切分数据，这也是随机的访问。这样会带来严重的系统性能下降。而直方图算法则是天然的cache friendly。在直方图算法的第3个for循环的时候，就已经统计好了每个bin的梯度，因此，在计算gain的时候，只需要对bin进行访问，造成的cache-miss问题会小很多。<br>
* 最后，在数据并行的时候，用 histgoram 可以大幅降低通信代价。用 pre-sorted 算法的话，通信代价是非常大的（几乎是没办法用的）。所以 xgoobst 在并行的时候也使用 histogram 进行通信。<br>


#### 劣势
* histogram 算法不能找到很精确的分割点，训练误差没有 pre-sorted 好。但从实验结果来看， histogram 算法在测试集的误差和 pre-sorted 算法差异并不是很大，甚至有时候效果更好。实际上可能决策树对于分割点的精确程度并不太敏感，而且较“粗”的分割点也自带正则化的效果，再加上boosting算法本身就是弱分类器的集成。<br>

参考资料：<br>
1. [机器学习算法梳理-LightGBM](https://blog.csdn.net/mingxiaod/article/details/86233309)<br>
2. [LightGBM算法总结](https://blog.csdn.net/weixin_39807102/article/details/81912566)<br>
<br>


## 4. leaf-wise VS level-wise
lightGBM模型采用带深度限制的Leaf-wise的叶子生长策略<br>

### level-wise
绝大多数的GBDT工具使用按层生长的决策树生长策略。level-wise过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。level-wise过程如下图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/level_wise.png)<br>

### leaf-wise
leaf-wise是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子节点，然后分裂。如此循环。同level-wise相比，在分裂次数相同的情况下，leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长处比较深的决策树，产生过拟合。因此LightGBM在leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。leaf-wise过程如下图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/leaf_wise.png)<br>

参考资料：<br>
1. [机器学习算法梳理-LightGBM](https://blog.csdn.net/mingxiaod/article/details/86233309)<br>
2. [LightGBM算法梳理](https://blog.csdn.net/qq_32577043/article/details/86215754#levelwise_37)<br>
<br>

## 5. 特征并行和数据并行

### 5.1 特征并行
特征并行的主要思想是：不同机器在不同的特征集合上分别寻找最优的分割点，然后再机器间同步最优的分割点，示意图如下：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/feature_para.png)<br>
<br>
传统算法中的特征并行，主要是体现在找到最好的分割点，其步骤为：<br>
1.垂直分区数据（不同的线程具有不同的数据集）；<br>
2.在本地数据集上找到最佳分割点，包括特征，阈值；<br>
3.再进行各个划分的通信整合并得到最佳划分；<br>
4.以最佳划分方法对数据进行划分，并将数据划分结果传递给其他线程；<br>
5.其他线程对接受到的数据进一步划分；<br><br>
传统特征并行的缺点：<br>
计算成本较大，传统特征并行没有实现得到"split"（时间复杂度为“O（训练样本的个数)"）的加速。当数据量很大的时候，难以加速；<br>
需要对划分的结果进行通信整合，其额外的时间复杂度约为 “O（训练样本的个数/8）”（一个数据一个字节）<br><br>

由于特征并行在训练样本的个数大的时候不能很好地加速，LightGBM做了以下优化：不是垂直分割数据，而是每个线程都拥有完整的全部数据。因此，因此最优的特征分裂结果不需要传输到其他线程，只需要将最优特征以及分裂点告诉其他线程，随后再本地进行处理。实际上这是一种牺牲空间换取时间的做法。<br><br>

处理过程如下：<br>
1.每个worker在基于局部的特征集合找到最优分裂特征。<br>
2.worker间传输最优分裂信息，并得到全局最优分裂信息。<br>
3.每个worker基于全局最优分裂信息，在本地进行数据分裂。<br><br>

### 5.2 数据并行
数据并行的思路如下图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/data_para.png)<br>
传统算法数据并行旨在并行化整个决策学习。数据并行的过程是：<br>
1、水平划分数据；<br>
2、线程以本地数据构建本地直方图；<br>
3、将本地直方图整合成全局直方图；<br>
4、在全局直方图中寻找最佳划分，然后执行此划分；<br><br>
传统数据并行的缺点：通信成本高。如果使用点对点通信算法，则一台机器的通信成本约为O(#machine * #feature * #bin)。如果使用聚合通信算法（例如“All Reduce”），通信成本约为O(2 * #feature * #bin)。<br><br>

LightGBM中通过下面方法来降低数据并行的通信成本：<br>
1、不同于“整合所有本地直方图以形成全局直方图”的方式，LightGBM 使用分散规约(Reduce scatter)的方式对不同线程的不同特征（不重叠的）进行整合。 然后线程从本地整合直方图中寻找最佳划分并同步到全局的最佳划分中；<br>
2、LightGBM通过直方图的减法加速训练。 基于此，我们可以进行单叶子节点的直方图通讯，并且在相邻直方图上作减法；<br><br>
通过上述方法，LightGBM 将数据并行中的通讯开销减少到O(0.5 * #feature * #bin)。<br><br>

### 5.3 投票并行
基于投票机制的并行算法，是在每个worker中选出top k个分裂特征，然后将每个worker选出的k个特征进行汇总，并选出全局分裂特征，进行数据分裂。有理论证明，这种voting parallel以很大的概率选出实际最优的特征，因此不用担心top k的问题，原理图如下：
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_LightGBM/vote_para.png)<br>


参考资料：<br>
1. [高级算法梳理之LightGBM](https://blog.csdn.net/sun_xiao_kai/article/details/90377282)<br>
2. [LightGBM算法梳理](https://blog.csdn.net/qq_32577043/article/details/86215754#levelwise_37)<br>
<br>


## 6. 顺序访问梯度
预排序算法中有两个频繁的操作会导致cache-miss，也就是缓存消失（对速度的影响很大，特别是数据量很大的时候，顺序访问比随机访问的速度快4倍以上）。<br><br>
对梯度的访问：在计算增益的时候需要利用梯度，对于不同的特征，访问梯度的顺序是不一样的，并且是随机的<br>
对于索引表的访问：预排序算法使用了行号和叶子节点号的索引表，防止数据切分的时候对所有的特征进行切分。同访问梯度一样，所有的特征都要通过访问这个索引表来索引。<br><br>
这两个操作都是随机的访问，会给系统性能带来非常大的下降。<br><br>
LightGBM使用的直方图算法能很好的解决这类问题。首先。对梯度的访问，因为不用对特征进行排序，同时，所有的特征都用同样的方式来访问，所以只需要对梯度访问的顺序进行重新排序，所有的特征都能连续的访问梯度。并且直方图算法不需要把数据id到叶子节点号上（不需要这个索引表，没有这个缓存消失问题）<br><br>

参考资料：<br>
1. [机器学习算法梳理-LightGBM](https://blog.csdn.net/mingxiaod/article/details/86233309)<br>
<br>

## 7. 支持类别特征
大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，转化到多维的0，1特征（one-hot操作），降低了空间和时间的效率。类别特征的使用时在实践中很常用的，基于这个考虑，LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的one-hot操作。<br>
LightGBM使用直方图的方式去处理，max bin的默认值是256，对于类别型特征值，每一个类别放入一个bin,当类别大于256时，会忽略那些很少出现的类别。在分裂的时候，算的是按“是否属于某个类别”来划分增益。实际效果就是类似于One-hot的编码方式。<br>

参考资料：<br>
1. [LightGBM算法梳理](https://blog.csdn.net/qq_32577043/article/details/86215754#levelwise_37)<br>
<br>

## 8. sklearn参数

参考资料：<br>
1. [LightGBM算法总结](https://blog.csdn.net/weixin_39807102/article/details/81912566)<br>
<br>

## 9. CatBoost

参考资料：<br>
1. [机器学习算法梳理-LightGBM](https://blog.csdn.net/mingxiaod/article/details/86233309)<br>
2. [LightGBM算法总结](https://blog.csdn.net/weixin_39807102/article/details/81912566)<br>
3. [XGB算法梳理](https://blog.csdn.net/wangrongrongwq/article/details/86755915#2.%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86)<br>
<br>







