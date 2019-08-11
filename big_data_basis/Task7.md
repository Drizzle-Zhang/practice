# 【Task7】实践

## 1. 计算每个content的CTR。

下载完成的数据格式如下：
```
uid     content_list    content_id
0       164423,430922,112513,485726,488385,340139,489273,391258 112513
1       635374,409237,586823,305055,519191,772121,788428,754213 305055,586823,305055,305055
2       57518,70020,828660,9511,477360,821209,178443,973485     178443,70020,178443,9511
3       542973,871389,914465,513667,536708,646545,90801,994236  536708
4       530817,401690,813927,107595,472415,375159,11354,281431  530817,375159
5       282200,402105,913036,389736,392579,166522,14420,314787  402105,282200,166522
6       568328,381531,873759,157884,812936,112027,602916,714218 381531,602916,568328,568328,112027,112027
```
计算目标是在每个uid中，计算content_list个数与content_id个数的商<br><br>

启动pyspark，运行下面的代码：
```Python
# 设置数据的路径
contentData = sc.textFile("file:///local/zy/download/content_list_id.txt")

# 计算每个uid的CTR
ctr = contentData.map(lambda x:x.split('\t')).map(lambda line:(line[0], len(line[1].split(','))/len(line[2].split(','))))

# 计算结果如下
>>> ctr.take(10)
[('uid', 1.0), ('0', 8.0), ('1', 2.0), ('2', 2.0), ('3', 8.0), ('4', 4.0), ('5', 2.6666666666666665), ('6', 1.3333333333333333), ('7', 2.0), ('8', 2.0)]

```

## 2. 【选做】 使用Spark实现ALS矩阵分解算法


## 3. 使用Spark分析Amazon DataSet(实现 Spark LR、Spark TFIDF)
[数据来源](http://jmcauley.ucsd.edu/data/amazon/)：<br>
下载files里面的"Home and Kitchen - 5-core"数据集<br><br>
**接下来先计算TF-IDF**<br>
读取json数据
```Python
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
home_kitchen = sqlContext.read.json("file:///local/zy/download/reviews_Home_and_Kitchen_5.json")
```
查看数据结构
```
>>> home_kitchen.printSchema()
root
 |-- asin: string (nullable = true)
 |-- helpful: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- overall: double (nullable = true)
 |-- reviewText: string (nullable = true)
 |-- reviewTime: string (nullable = true)
 |-- reviewerID: string (nullable = true)
 |-- reviewerName: string (nullable = true)
 |-- summary: string (nullable = true)
 |-- unixReviewTime: long (nullable = true)

>>> home_kitchen.first()
Row(asin='0615391206', helpful=[0, 0], overall=5.0, reviewText='My daughter wanted this book and the price on Amazon was the best.  She has already tried one recipe a day after receiving the book.  She seems happy with it.', reviewTime='10 19, 2013', reviewerID='APYOBQE6M18AA', reviewerName='Martin Schwartz', summary='Best Price', unixReviewTime=1382140800)

```
我们只需要用text数据
```Python
>>> df = home_kitchen.select('reviewText')
```
加上一列id，以便统计评论个数：
```Python
from pyspark.sql import functions as F
df = df.withColumn("doc_id", F.monotonically_increasing_id())
```
使用空格进行分词
```Python
df = df.withColumn('keys',F.split('reviewText', " ")).drop('reviewText')
```
然后把分好的词explode一下，这样每个评论及其每个单词都会形成一行
```Python
NUM_doc = df.count()
# One hot words
df = df.select('*', F.explode('keys').alias('token'))
```
计算TF，TF是针对一篇文章而言的，是一篇文章中的单词频数/单词总数，这里的一篇文章就是一条评论。
```Python
# Calculate TF
TF = df.groupBy("doc_id").agg(F.count("token").alias("doc_len")) \
    .join(df.groupBy("doc_id", "token")
          .agg(F.count("keys").alias("word_count")), ['doc_id']) \
    .withColumn("tf", F.col("word_count") / F.col("doc_len")) \
    .drop("doc_len", "word_count")
TF.cache()
```
这里以评论id分组，并计算每个组内单词的个数，也就是每个评论有多少单词（doc_len），然后和另一个df2以字段“doc_id”内连接，df2以评论id和单词分组，计算组内分词集合的个数，也就是每个词出现在多少集合中（word_count）。最后再添加一列tf值，即单词在文档中出现的次数/文档总词数。<br><br>

计算IDF，IDF是逆文档频率，表示一个单词在语料库中出现的频率，也就是一个单词在多少篇文章中出现了。
```Python
# Calculate IDF
IDF = df.groupBy("token").agg(F.countDistinct("doc_id").alias("df"))
IDF = IDF.select('*', (F.log(NUM_doc / (IDF['df'] + 1))).alias('idf'))
IDF.cache()
```
这里以每个单词分组，计算单词在不同评论中出现的次数，然后再用log(训练语料的总文档数/(出现词语x的文档数+1)）计算出idf值。<br><br>

计算TF-IDF，两个df以单词为字段join，得到TF-IDF值。
```Python
# Calculate TF-IDF
TFIDF = TF.join(IDF, ['token']).withColumn('tf-idf', F.col('tf') * F.col('idf'))
```


**Reference:**<br>
1. [【Spark机器学习速成宝典】模型篇03线性回归【LR】（Python版） ](https://www.cnblogs.com/itmorn/p/8023396.html)<br>
2. [python spark 简单的gbdt, LR模型的使用示例](https://blog.csdn.net/qq_36480160/article/details/82013975)<br>
3. [Spark 2.1.0 入门：特征抽取 — TF-IDF(Python版)](http://dblab.xmu.edu.cn/blog/1766-2/)<br>
4. [PySpark TF-IDF计算（2）](https://blog.csdn.net/macanv/article/details/87731785)<br>
5. [【干货】Python大数据处理库PySpark实战——使用PySpark处理文本多分类问题](https://cloud.tencent.com/developer/article/1096712)<br>
6. [利用Spark计算TF-IDF ](https://fuhailin.github.io/Calculating-TF-IDF-With-Apache-Spark/)<br>

<br>









