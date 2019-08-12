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
在这里，l是一个可微的凸损失函数，用来度量预测值和真实值之间的差距。第二项ohm惩罚模型的复杂度。这个附加的正则项有助于平滑最终的学习权重以防止过拟合。<br>
接下来的推导过程，实质上是对某一轮迭代中，其中的某一棵树进行讨论的：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/loss_function1.png)<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/loss_function2.png)<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/loss_function3.png)<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/loss_function4.png)<br>

参考资料：<br>
1. [XGB简介](https://www.jianshu.com/p/3d5a4dcb3ae4)<br>
2. [机器学习算法梳理—XGB](https://blog.csdn.net/mingxiaod/article/details/86063153)<br>
3. [XGB算法梳理](https://blog.csdn.net/wangrongrongwq/article/details/86755915#2.%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86)<br>
<br>


## 3. 分裂结点算法
### 树结构的学习
理论上来说，需要对所有可能的树结构q进行枚举，选出最优的q，但是这个计算量是不可承受的。因此，可以使用贪心算法，从一个只有一个叶子的树开始，往树上迭代进行分支，这就用得上下面的公式了：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/split1.png)<br>
该公式的作用类似于基尼系数或是信息增益，用来确定节点是否应该分裂<br>
得到该公式的推导过程如下：<br>
![](http://latex.codecogs.com/gif.latex?\$$Obj_{split}=-\frac{1}{2}[\sum^{T_{split}-2}_{j=1}{\frac{G^{2}_{j}}{H^{2}_{j}+\lambda}}+\frac{G^{2}_{L}}{H^{2}_{R}+\lambda}+\frac{G^{2}_{R}}{H^{2}_{R}+\lambda}]+T_{split}\cdot\gamma$$)<br>
![](http://latex.codecogs.com/gif.latex?\$$Obj_{nosplit}=-\frac{1}{2}[\sum^{T_{nosplit}-1}_{j=1}{\frac{G^{2}_{j}}{H^{2}_{j}+\lambda}}+\frac{(G_{L}+G_{R})^{2}}{H_{L}+H_{R}+\lambda}]+T_{nosplit}\cdot\gamma$$)<br>
![](http://latex.codecogs.com/gif.latex?\$$Gain=Obj_{nosplit}-Obj_{split}=\frac{1}{2}[\frac{G^{2}_{L}}{H^{2}_{R}+\lambda}+\frac{G^{2}_{R}}{H^{2}_{R}+\lambda}-\frac{(G_{L}+G_{R})^{2}}{H_{L}+H_{R}+\lambda}]-\gamma$$)<br>

### 贪心算法
贪心算法就是对于每一棵树，在每次进行分支时，使用上面计算Gain的公式，对所有特征计算都一遍，选出最合适的分割点。某一次节点分裂的计算过程如下：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/split2.png)<br>

### 近似算法
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/split3.png)<br>

## 4. 正则化以及防止过拟合的方法
上面介绍算法原理部分已经指出损失函数中带有正则项，但是XGB中还有一些防止过拟合的方法。<br>
其中一种是Shrinkage方法，这种方法会在每轮迭代中的叶子节点分数wj上增加一个缩减因子，这样会减少每棵单独的树和其叶子节点对未来的树的影响。<br>
另外一种方法是Column Subsampling，这中对特征进行子采样的方法之前在RF中已经介绍过，它是每次建树时抽取一部分特征来建树，这里选择特征可以是根据GINI指数来选择，即选择GINI指数较大（即信息量最大，最不纯的特征）的一些特征来建树（当然这里前提是基分类器是CART的情况）他可以起到防止过拟合作用，甚至还有助于加快并行算法的运行速度。<br><br>

## 5. 对缺失值处理
有很多种原因可能导致特征的稀疏（缺失），所以当遇到样本某个维度的特征缺失的时候，就不能知道这个样本会落在左子节点还是右子节点。xgboost把缺失值当做稀疏矩阵来对待，本身在节点分裂时不考虑缺失值的数值，但确定分裂的特征后，样本落在哪个子节点得分高，就放到哪里。如果训练中没有数据缺失，预测时出现了数据缺失，那么默认被分类到右子树。具体的算法如下：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/ensemble_learning/Supp_Task3/missing1.png)<br>


## 6. 优缺点
xgBoosting在传统Boosting的基础上，利用cpu的多线程，引入正则化项，加入剪纸，控制了模型的复杂度。<br>
与GBDT相比，xgBoosting有以下进步：<br>
1）GBDT以传统CART作为基分类器，而xgBoosting支持线性分类器，相当于引入L1和L2正则化项的逻辑回归（分类问题）和线性回归（回归问题）；<br>
2）GBDT在优化时只用到一阶导数，xgBoosting对代价函数做了二阶Talor展开，引入了一阶导数和二阶导数；<br>
3）当样本存在缺失值是，xgBoosting能自动学习分裂方向；<br>
4）xgBoosting借鉴RF的做法，支持列抽样，这样不仅能防止过拟合，还能降低计算；<br>
5）xgBoosting的代价函数引入正则化项，控制了模型的复杂度，正则化项包含全部叶子节点的个数，每个叶子节点输出的score的L2模的平方和。从贝叶斯方差角度考虑，正则项降低了模型的方差，防止模型过拟合；<br>
6）xgBoosting在每次迭代之后，为叶子结点分配学习速率，降低每棵树的权重，减少每棵树的影响，为后面提供更好的学习空间；<br>
7）xgBoosting工具支持并行,但并不是tree粒度上的，而是特征粒度，决策树最耗时的步骤是对特征的值排序，xgBoosting在迭代之前，先进行预排序，存为block结构，每次迭代，重复使用该结构，降低了模型的计算；block结构也为模型提供了并行可能，在进行结点的分裂时，计算每个特征的增益，选增益最大的特征进行下一步分裂，那么各个特征的增益可以开多线程进行；<br>
8）可并行的近似直方图算法，树结点在进行分裂时，需要计算每个节点的增益，若数据量较大，对所有节点的特征进行排序，遍历的得到最优分割点，这种贪心法异常耗时，这时引进近似直方图算法，用于生成高效的分割点，即用分裂后的某种值减去分裂前的某种值，获得增益，为了限制树的增长，引入阈值，当增益大于阈值时，进行分裂；<br>
然而，与LightGBM相比，又表现出了明显的不足：<br>
1）xgBoosting采用预排序，在迭代之前，对结点的特征做预排序，遍历选择最优分割点，数据量大时，贪心法耗时，LightGBM方法采用histogram算法，占用的内存低，数据分割的复杂度更低；<br>
2）xgBoosting采用level-wise生成决策树，同时分裂同一层的叶子，从而进行多线程优化，不容易过拟合，但很多叶子节点的分裂增益较低，没必要进行跟进一步的分裂，这就带来了不必要的开销；LightGBM采用深度优化，leaf-wise生长策略，每次从当前叶子中选择增益最大的结点进行分裂，循环迭代，但会生长出更深的决策树，产生过拟合，因此引入了一个阈值进行限制，防止过拟合.<br>
<br>

## 7. 应用场景


## 8. sklearn参数
在运行XGBoost程序之前，必须设置三种类型的参数：通用类型参数（general parameters）、booster参数和学习任务参数（task parameters）。<br>
　　一般类型参数general parameters –参数决定在提升的过程中用哪种booster，常见的booster有树模型和线性模型。<br>
　　Booster参数-该参数的设置依赖于我们选择哪一种booster模型。<br>
　　学习任务参数task parameters-参数的设置决定着哪一种学习场景，例如，回归任务会使用不同的参数来控制着排序任务。<br>
　　命令行参数-一般和xgboost的CL版本相关。<br>
Booster参数：<br>
　　1. eta[默认是0.3]  和GBM中的learning rate参数类似。通过减少每一步的权重，可以提高模型的鲁棒性。典型值0.01-0.2<br>
　　2. min_child_weight[默认是1]  决定最小叶子节点样本权重和。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合。这个参数需要用cv来调整<br>
　　3. max_depth [默认是6]  树的最大深度，这个值也是用来避免过拟合的3-10<br>
　　4. max_leaf_nodes  树上最大的节点或叶子的数量，可以代替max_depth的作用，应为如果生成的是二叉树，一个深度为n的树最多生成2n个叶子,如果定义了这个参数max_depth会被忽略<br>
　　5. gamma[默认是0]  在节点分裂时，只有在分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数值越大，算法越保守。<br>
　　6. max_delta_step[默认是0]  这参数限制每颗树权重改变的最大步长。如果是0意味着没有约束。如果是正值那么这个算法会更保守，通常不需要设置。<br>
　　7. subsample[默认是1]  这个参数控制对于每棵树，随机采样的比例。减小这个参数的值算法会更加保守，避免过拟合。但是这个值设置的过小，它可能会导致欠拟合。典型值：0.5-1<br>
　　8. colsample_bytree[默认是1]  用来控制每颗树随机采样的列数的占比每一列是一个特征0.5-1<br>
　　9. colsample_bylevel[默认是1]  用来控制的每一级的每一次分裂，对列数的采样的占比。<br>
　　10. lambda[默认是1]  权重的L2正则化项<br>
　　11. alpha[默认是1]  权重的L1正则化项<br>
　　12. scale_pos_weight[默认是1]  各类样本十分不平衡时，把这个参数设置为一个正数，可以使算法更快收敛。<br>
通用参数：<br>
　　1． booster[默认是gbtree]<br>
　　选择每次迭代的模型，有两种选择：gbtree基于树的模型、gbliner线性模型<br>
　　2． silent[默认是0]<br>
　　当这个参数值为1的时候，静默模式开启，不会输出任何信息。一般这个参数保持默认的0，这样可以帮我们更好的理解模型。<br>
　　3． nthread[默认值为最大可能的线程数]<br>
　　这个参数用来进行多线程控制，应当输入系统的核数，如果你希望使用cpu全部的核，就不要输入这个参数，算法会自动检测。<br><br>



