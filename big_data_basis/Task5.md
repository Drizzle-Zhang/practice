# 【Task5】Spark常用API

## 1. spark集群搭建
解压并移动到相应目录
```Bash
tar -zxvf spark-2.1.1-bin-hadoop2.7.tgz
```
修改/etc/profie，增加如下内容：
```
export SPARK_HOME=/local/zy/tools/spark-2.1.1-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
```
复制spark-env.sh.template成spark-env.sh，并添加以下内容
```
[zy@node7 spark-2.1.1-bin-hadoop2.7]$ cd conf/
[zy@node7 conf]$ ls
docker.properties.template  metrics.properties.template   spark-env.sh.template
fairscheduler.xml.template  slaves.template
log4j.properties.template   spark-defaults.conf.template
[zy@node7 conf]$ cp spark-env.sh.template spark-env.sh
[zy@node7 conf]$ vi spark-env.sh
###
export HADOOP_HOME=/local/zy/tools/hadoop-2.7.3
export HADOOP_CONF_DIR=/local/zy/tools/hadoop-2.7.3/etc/hadoop
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.151-1.b12.el7_4.x86_64/jre
# 设置Master的主机名
export SPARK_MASTER_IP=10.152.255.52
export SPARK_LOCAL_IP=10.152.255.52
export SPARK_MASTER_HOST=10.152.255.52
# 提交Application的端口，默认就是这个，万一要改呢，改这里
export SPARK_MASTER_PORT=7077
# 每一个Worker最多可以使用的cpu core的个数，我虚拟机就一个;真实服务器如果有32个，你>可以设置为32个
export SPARK_WORKER_CORES=20
# 每一个Worker最多可以使用的内存，我的虚拟机就2g
# 真实服务器如果有128G，你可以设置为100G
export SPARK_WORKER_MEMORY=10g
#export SCALA_HOME=/home/heitao/Soft/scala-2.11.0
export SPARK_HOME=/local/zy/tools/spark-2.1.1-bin-hadoop2.7
#export SPARK_DIST_CLASSPATH=$(/home/heitao/Soft/hadoop-2.7.3/bin/hadoop classpath)
###
```
复制slaves.template成slaves，并添加以下内容
```Bash
cp slaves.template slaves
###
node7
node8
node9
###
```
将配置好的spark文件复制到Slave1和Slave2节点。
```Bash
scp -r ./spark-2.1.1-bin-hadoop2.7 zy@10.152.255.53:/local/zy/tools
scp -r ./spark-2.1.1-bin-hadoop2.7 zy@10.152.255.54:/local/zy/tools
```
然后再修改其他节点的bashrc文件,增加Spark的配置，过程同Master一样<br>
再修改spark-env.sh，将export SPARK_LOCAL_IP=xxx.xxx.xxx.xxx改成node8和node9对应节点的IP。<br><br>
在Master节点启动集群
```Bash
[zy@node7 spark-2.1.1-bin-hadoop2.7]$ ./sbin/start-all.sh 
```
查看集群是否启动成功
```Bash[zy@node7 spark-2.1.1-bin-hadoop2.7]$ jps
19408 SecondaryNameNode
19121 NameNode
20722 ResourceManager
177040 Jps
176721 Worker
176661 Worker
176429 Master
[zy@node8 ~]$ jps
4881 Jps
183451 DataNode
184153 NodeManager
4655 Worker
```
node7在Hadoop的基础上新增了：Master<br>
node8,node9在Hadoop的基础上新增了：Worker<br><br>

**Reference:**<br>
1. [Spark伪分布式环境搭建 + jupyter连接spark集群 ](https://mp.weixin.qq.com/s?__biz=MzI3Mjg1OTA3NQ==&mid=2247483893&idx=1&sn=84496036abf5c302806f2daa9655bd6a&chksm=eb2d6b59dc5ae24fae6d483547778fc7bbe1054094ca97b4c152fdd79bbc8b0ed3e14ffefbfd&mpshare=1&scene=1&srcid=&sharer_sharetime=1564900973740&sharer_shareid=8ac76a2e8d1b620817577ca68d2d215f&key=1a2eded5d1d2d5f7e5ff6b1d226694b4aed37afd13c5472df38cb96541ce31181690e47492a199c87dcdff8410f92dd3c4fb25dc3c00b4c5dba4e30a0ef826d8c81f4db1037d3fbcffbc2f31f3cdb1ef&ascene=1&uin=MTExNjkzNDEwNg%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=Ika06k3RtS5H%2BXm0gmTpvebwTIuC5uQymoAZxQ6aQvMyKbEjXvF2WCwOqYhWuCiN)<br><br>


## 2. 初步认识Spark （解决什么问题，为什么比Hadoop快，基本组件及架构Driver/）
### 定义
Spark是Apache的一个顶级项目，是一个快速、通用的大规模数据处理引擎

### Spark比Hadoop快的主要原因
**1.消除了冗余的HDFS读写**<br>
Hadoop每次shuffle操作后，必须写到磁盘，而Spark在shuffle后不一定落盘，可以cache到内存中，以便迭代时使用。如果操作复杂，很多的shufle操作，那么Hadoop的读写IO时间会大大增加。<br>
**2.消除了冗余的MapReduce阶段**<br>
Hadoop的shuffle操作一定连着完整的MapReduce操作，冗余繁琐。而Spark基于RDD提供了丰富的算子操作，且reduce操作产生shuffle数据，可以缓存在内存中。<br>
**3.JVM的优化**<br>
Spark Task的启动时间快。Spark采用fork线程的方式，Spark每次MapReduce操作是基于线程的，只在启动。而Hadoop采用创建新的进程的方式，启动一个Task便会启动一次JVM。Spark的Executor是启动一次JVM，内存的Task操作是在线程池内线程复用的。每次启动JVM的时间可能就需要几秒甚至十几秒，那么当Task多了，这个时间Hadoop不知道比Spark慢了多少。<br>

### Spark基本组件
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/supp_Task5/elements.png)<br>
核心部分是RDD相关的，就是我们前面介绍的任务调度的架构，后面会做更加详细的说明。<br>

**SparkStreaming：**<br>
基于SparkCore实现的可扩展、高吞吐、高可靠性的实时数据流处理。支持从Kafka、Flume等数据源处理后存储到HDFS、DataBase、Dashboard中。<br>

**MLlib：**<br>
关于机器学习的实现库。<br>

**SparkSQL：**<br>
Spark提供的sql形式的对接Hive、JDBC、HBase等各种数据渠道的API，用Java开发人员的思想来讲就是面向接口、解耦合，ORMapping、Spring Cloud Stream等都是类似的思想。<br>

**GraphX：**<br>
关于图和图并行计算的API。<br>
 
**RDD(Resilient Distributed Datasets) 弹性分布式数据集**<br>
RDD支持两种操作：转换（transiformation）和动作（action）<br>
转换就是将现有的数据集创建出新的数据集，像Map；动作就是对数据集进行计算并将结果返回给Driver，像Reduce。<br>
RDD中转换是惰性的，只有当动作出现时才会做真正运行。这样设计可以让Spark更见有效的运行，因为我们只需要把动作要的结果送给Driver就可以了而不是整个巨大的中间数据集。<br>
缓存技术（不仅限内存，还可以是磁盘、分布式组件等）是Spark构建迭代式算法和快速交互式查询的关键，当持久化一个RDD后每个节点都会把计算分片结果保存在缓存中，并对此数据集进行的其它动作（action）中重用，这就会使后续的动作（action）变得跟迅速（经验值10倍）。例如RDD0àRDD1àRDD2，执行结束后RDD1和RDD2的结果已经在内存中了，此时如果又来RDD0àRDD1àRDD3，就可以只计算最后一步了。<br>



### Spark架构
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/supp_Task5/structure.png)<br>
**ClusterManage**r<br>
负责分配资源，有点像YARN中ResourceManager那个角色，大管家握有所有的干活的资源，属于乙方的总包。<br>

**WorkerNode**<br>
是可以干活的节点，听大管家ClusterManager差遣，是真正有资源干活的主。<br>

**Executor**<br>
是在WorkerNode上起的一个进程，相当于一个包工头，负责准备Task环境和执行Task，负责内存和磁盘的使用。<br>

**Task**<br>
是施工项目里的每一个具体的任务。<br>

**Driver**<br>
是统管Task的产生与发送给Executor的，是甲方的司令员。<br>

**SparkContext**<br>
是与ClusterManager打交道的，负责给钱申请资源的，是甲方的接口人。<br>


整个互动流程是这样的：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/supp_Task5/structure2.jpeg)<br>
1 甲方来了个项目，创建了SparkContext，SparkContext去找ClusterManager申请资源同时给出报价，需要多少CPU和内存等资源。ClusterManager去找WorkerNode并启动Excutor，并介绍Excutor给Driver认识。<br>

2 Driver根据施工图拆分一批批的Task，将Task送给Executor去执行。<br>

3 Executor接收到Task后准备Task运行时依赖并执行，并将执行结果返回给Driver。<br>

4 Driver会根据返回回来的Task状态不断的指挥下一步工作，直到所有Task执行结束。<br>


**Reference:**<br>
1. [Spark基础全解析](https://blog.csdn.net/vinfly_li/article/details/79396821)<br>
2. [spark为什么比hadoop的mr要快？(https://www.cnblogs.com/wqbin/p/10217994.html)<br>
3. [Spark原理详解](https://blog.csdn.net/yejingtao703/article/details/79438572)<br>
4. [spark核心技术原理透视一（Spark运行原理）](https://blog.csdn.net/liuxiangke0210/article/details/79687240)<br>
5. [Spark原理小总结](https://www.cnblogs.com/atomicbomb/p/7488278.html)<br><br>


## 3. 理解spark的RDD

RDD是Spark的基本抽象，是一个弹性分布式数据集，代表着不可变的，分区（partition）的集合，能够进行并行计算。也即是说：<br>
1. 它是一系列的分片、比如说128M一片，类似于Hadoop的split；<br>
2. 在每个分片上都有一个函数去执行/迭代/计算它<br>
3. 它也是一系列的依赖，比如RDD1转换为RDD2，RDD2转换为RDD3，那么RDD2依赖于RDD1，RDD3依赖于RDD2。<br>
4. 对于一个Key-Value形式的RDD，可以指定一个partitioner，告诉它如何分片，常用的有hash、range<br>
5. 可选择指定分区最佳计算位置<br>

**Reference:**<br>
[Spark基础全解析](https://blog.csdn.net/vinfly_li/article/details/79396821)<br><br>


## 4. 使用shell方式操作Spark，熟悉RDD的基本操作

RDD的操作分为两种，一种是转化操作，一种是执行操作，转化操作并不会立即执行，而是到了执行操作才会被执行<br>

### 转化操作：

`map()` 参数是函数，函数应用于RDD每一个元素，返回值是新的RDD<br>
`flatMap()` 参数是函数，函数应用于RDD每一个元素，将元素数据进行拆分，变成迭代器，返回值是新的RDD<br>
`filter()` 参数是函数，函数会过滤掉不符合条件的元素，返回值是新的RDD<br>
`distinct()` 没有参数，将RDD里的元素进行去重操作<br>
`union()` 参数是RDD，生成包含两个RDD所有元素的新RDD<br>
`intersection()` 参数是RDD，求出两个RDD的共同元素<br>
`subtract()` 参数是RDD，将原RDD里和参数RDD里相同的元素去掉<br>
`cartesian()` 参数是RDD，求两个RDD的笛卡儿积<br>

### 行动操作：

`collect()` 返回RDD所有元素<br>
`count()` RDD里元素个数<br>
`countByValue()` 各元素在RDD中出现次数<br>
`reduce()` 并行整合所有RDD数据，例如求和操作<br>
`fold(0)(func)` 和reduce功能一样，不过fold带有初始值<br>
`aggregate(0)(seqOp,combop)` 和reduce功能一样，但是返回的RDD数据类型和原RDD不一样<br>
`foreach(func)` 对RDD每个元素都是使用特定函数<br>

行动操作每次的调用时不会存储前面的计算结果的，若果想要存储前面的操作结果需要把结果加载需要在需要缓存中间结果的RDD调用`cache()`,cache()方法是把中间结果缓存到内存中，也可以指定缓存到磁盘中（也可以只用`persisit()`）<br>


**Reference:**<br>
1. [spark-shell基本的RDD操作](https://blog.csdn.net/vinfly_li/article/details/79396821)<br>
2. [Spark基础全解析](https://blog.csdn.net/vinfly_li/article/details/79396821)<br><br>


##  5. 使用jupyter连接集群的pyspark
先在node7上进行操作<br>
先备份系统自带的python，然后安装Anaconda
```Bash
[zy@node7 tools]$ ll /usr/bin/python
lrwxrwxrwx 1 root root 7 May 27  2017 /usr/bin/python -> python2
[root@node7 tools]# mv /usr/bin/python /usr/bin/python.bak
[root@node7 tools]# sh Anaconda3-2019.07-Linux-x86_64.sh
```
生成配置文件
```Bash
[zy@node7 tools]$ jupyter notebook --generate-config
Writing default config to: /local/zy/.jupyter/jupyter_notebook_config.py
```
打开python窗口，并输入
```Python
[zy@node7 tools]$ python
Python 3.7.3 (default, Mar 27 2019, 22:11:17) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from notebook.auth import passwd
>>> passwd()
Enter password: 
Verify password: 
'sha1:182bc6e73e06:efa239a9d2e45947bfdd1da9472afd3e604e77f1'
```
修改配置文件
```
vim ~/.jupyter/jupyter_notebook_config.py
###
## The IP address the notebook server will listen on.
c.NotebookApp.ip = '0.0.0.0' # 所有IP可访问

#  The string should be of the form type:salt:hashed-password.
c.NotebookApp.password = 'sha1:182bc6e73e06:efa239a9d2e45947bfdd1da9472afd3e604e77f1' # 刚刚生成的密匙

#  configuration option.
c.NotebookApp.open_browser = False # 禁止自动打开浏览器

## The port the notebook server will listen on.
c.NotebookApp.port = 8888  # 指定８８８８端口

###
```
远程端启动jupyter
```Bash
[zy@node7 tools]$ jupyter notebook
```
在浏览器上输入localhost:8888也可以远程访问jupyter<br><br>

接下来使用jupyter链接spark集群，先安装pyspark包
```
pip install pypandoc py4j pyspark
```
配置环境变量
```
export PYSPARK_DRIVER_PYTHON=/local/zy/tools/anaconda3/bin/jupyter-notebook
export PYSPARK_DRIVER_PYTHON_OPTS="--ip=0.0.0.0 --port=8888"
```
远程启动pyspark
```
[zy@node7 tools]$ pyspark
[I 13:07:48.657 NotebookApp] JupyterLab extension loaded from /local/zy/tools/anaconda3/lib/python3.7/site-packages/jupyterlab
[I 13:07:48.657 NotebookApp] JupyterLab application directory is /local/zy/tools/anaconda3/share/jupyter/lab
[I 13:07:48.659 NotebookApp] Serving notebooks from local directory: /local/zy/tools
[I 13:07:48.659 NotebookApp] The Jupyter Notebook is running at:
[I 13:07:48.659 NotebookApp] http://node7:8888/
[I 13:07:48.659 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

```
在本地浏览器输入localhost:8888就可以访问远程服务器上的jupyter了


**Reference:**<br>
1. [Spark伪分布式环境搭建 + jupyter连接spark集群 ](https://mp.weixin.qq.com/s?__biz=MzI3Mjg1OTA3NQ==&mid=2247483893&idx=1&sn=84496036abf5c302806f2daa9655bd6a&chksm=eb2d6b59dc5ae24fae6d483547778fc7bbe1054094ca97b4c152fdd79bbc8b0ed3e14ffefbfd&mpshare=1&scene=1&srcid=&sharer_sharetime=1564900973740&sharer_shareid=8ac76a2e8d1b620817577ca68d2d215f&key=1a2eded5d1d2d5f7e5ff6b1d226694b4aed37afd13c5472df38cb96541ce31181690e47492a199c87dcdff8410f92dd3c4fb25dc3c00b4c5dba4e30a0ef826d8c81f4db1037d3fbcffbc2f31f3cdb1ef&ascene=1&uin=MTExNjkzNDEwNg%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=Ika06k3RtS5H%2BXm0gmTpvebwTIuC5uQymoAZxQ6aQvMyKbEjXvF2WCwOqYhWuCiN)<br><br>


## 6. 理解Spark的shuffle过程
### Shuffle的作用
Shuffle的中文解释为“洗牌操作”，可以理解成将集群中所有节点上的数据进行重新整合分类的过程。其思想来源于hadoop的mapReduce,Shuffle是连接map阶段和reduce阶段的桥梁。由于分布式计算中，每个阶段的各个计算节点只处理任务的一部分数据，若下一个阶段需要依赖前面阶段的所有计算结果时，则需要对前面阶段的所有计算结果进行重新整合和分类，这就需要经历shuffle过程。
在spark中，RDD之间的关系包含窄依赖和宽依赖，其中宽依赖涉及shuffle操作。因此在spark程序的每个job中，都是根据是否有shuffle操作进行阶段（stage）划分，每个stage都是一系列的RDD map操作。<br>

### shuffle操作为什么耗时
shuffle操作需要将数据进行重新聚合和划分，然后分配到集群的各个节点上进行下一个stage操作，这里会涉及集群不同节点间的大量数据交换。由于不同节点间的数据通过网络进行传输时需要先将数据写入磁盘，因此集群中每个节点均有大量的文件读写操作，从而导致shuffle操作十分耗时（相对于map操作）。<br>

### Spark目前的ShuffleManage模式及处理机制
Spark程序中的Shuffle操作是通过shuffleManage对象进行管理。Spark目前支持的ShuffleMange模式主要有两种：HashShuffleMagnage 和SortShuffleManage
Shuffle操作包含当前阶段的Shuffle Write（存盘）和下一阶段的Shuffle Read（fetch）,两种模式的主要差异是在Shuffle Write阶段，下面将着重介绍。<br>

**Reference:**<br>
1. [Spark 的Shuffle过程详解](https://blog.csdn.net/zylove2010/article/details/79067149)<br>
2. [彻底搞懂spark的shuffle过程（shuffle write）](https://www.cnblogs.com/itboys/p/9201750.html)<br>
3. [关于spark shuffle过程的理解](https://blog.csdn.net/quitozang/article/details/80904040)<br><br>


## 7. 学会使用SparkStreaming

### Spark Streaming程序基本步骤
1.通过创建输入DStream来定义输入源<br>
2.通过对DStream应用转换操作和输出操作来定义流计算。<br>
3.用streamingContext.start()来开始接收数据和处理流程。<br>
4.通过streamingContext.awaitTermination()方法来等待处理结束（手动结束或因为错误而结束）。<br>
5.可以通过streamingContext.stop()来手动结束流计算进程。<br><br>

### 创建StreamingContext对象
请登录Linux系统，启动pyspark。进入pyspark以后，就已经获得了一个默认的SparkConext，也就是sc。因此，可以采用如下方式来创建StreamingContext对象：
```Python
>>> from pyspark import SparkContext
>>> from pyspark.streaming import StreamingContext
>>> ssc = StreamingContext(sc, 1)
```
1表示每隔1秒钟就自动执行一次流计算，这个秒数可以自由设定。<br>
如果是编写一个独立的Spark Streaming程序，而不是在pyspark中运行，则需要通过如下方式创建StreamingContext对象：
```Python
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
conf = SparkConf()
conf.setAppName('TestDStream') # 设置应用程序名称
conf.setMaster('local[2]') # local[2]表示本地模式，启动2个工作线程
sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 1)
```

### 文件流(DStream) - 命令行中监听
Spark支持从兼容HDFS API的文件系统中读取数据，创建数据流。<br>
为了能够演示文件流的创建，我们需要首先创建一个日志目录，并在里面放置两个模拟的日志文件。log1.txt输入：
```
I love Hadoop
I love Spark
Spark is fast
```
请另外打开一个终端窗口，启动进入pyspark
```
>>> from operator import add
>>> from pyspark import SparkContext
>>> from pyspark.streaming import StreamingContext
>>> ssc = StreamingContext(sc,20)
>>> lines = ssc.textFileStream('/local/zy/spark/logfile')
>>> words = lines.flatMap(lambda line:line.split(' '))
>>> wordCounts = words.map(lambda word:(word,1)).reduceByKey(add)
>>> wordCounts.pprint()
>>> ssc.start()
>>> ssc.awaitTermination()
>>> ssc.awaitTermination()
```
输入ssc.start()以后，程序就开始自动进入循环监听状态，屏幕上会显示一堆的信息,下面的ssc.awaitTermination()是无法输入到屏幕上的。<br>
Spark Streaming每隔20秒就监听一次。但是，监听程序只监听日志目录下在程序启动后新增的文件，不会去处理历史上已经存在的文件。所以，为了让我们能够看到效果，需要到日志目录下再新建一个log3.txt文件。请打开另外一个终端窗口再新建一个log3.txt文件，里面随便输入一些英文单词，保存后再切换回到spark-shell窗口。<br>
现在你会发现屏幕上不断输出新的信息，导致你无法看清。所以必须停止这个监听程序，按键盘Ctrl+D，或者Ctrl+C。<br>
你可以看到屏幕上，在一大堆输出信息中，你可以找到打印出来的单词统计信息。<br>


**Reference:**<br>
1. [SparkStreaming教程](https://www.jianshu.com/p/f11e6611bc7a)<br>
2. [Spark学习(Python版本)：SparkStreaming基本操作](https://www.jianshu.com/p/66d3914f4cf1)<br><br>


## 8. 说一说take,collect,first的区别，为什么不建议使用collect？

first: 返回第一个元素 
```
scala> val rdd = sc.parallelize(List(1,2,3,3))
 
scala> rdd.first()
res1: Int = 1
```
take: rdd.take(n)返回第n个元素 
```
scala> val rdd = sc.parallelize(List(1,2,3,3))
 
scala> rdd.take(2)
res3: Array[Int] = Array(1, 2)
```
collect: rdd.collect() 返回 RDD 中的所有元素 
```
scala> val rdd = sc.parallelize(List(1,2,3,3))
 
scala> rdd.collect()
res4: Array[Int] = Array(1, 2, 3, 3)
```
如果数据量比较大的时候，尽量不要使用collect函数，因为这可能导致Driver端内存溢出问题。


**Reference:**<br>
1. [spark RDD算子（九）之基本的Action操作 first, take, collect, count, countByValue, reduce, aggregate, fold,top](https://blog.csdn.net/t1dmzks/article/details/70667011)<br>
2. [Spark函数讲解：collect](https://blog.csdn.net/LW_GHY/article/details/51477130)<br><br>


## 9. 向集群提交Spark程序
### 启动Spark集群
请登录Linux系统，打开一个终端，启动Hadoop集群；然后启动Spark的Master节点和所有slaves节点
```Shell
$HADOOP_HOME/sbin/start-all.sh
cd $SPARK_HOME
sbin/start-master.sh
sbin/start-slaves.sh
```
### 独立集群管理器
（1）在集群中运行应用程序JAR包<br>
向独立集群管理器提交应用，需要把spark：//master:7077作为主节点参数递给spark-submit。下面我们可以运行Spark安装好以后自带的样例程序SparkPi，它的功能是计算得到pi的值（3.1415926）。<br>
在Shell中输入如下命令：
```Shell
bin/spark-submit --class org.apache.spark.examples.SparkPi --master spark://master:7077 examples/jars/spark-examples_2.11-2.0.2.jar 100 2>&1 | grep "Pi is roughly"
```
（2）在集群中运行pyspark
```Shell
pyspark
```
可以在pyspark中输入如下代码进行测试：
```Python
>>> textFile = sc.textFile("hdfs://master:9000/README.md")
>>> textFile.count()
99                                                                 
>>> textFile.first()
# Apache Spark
```

### Hadoop YARN管理器
（1）在集群中运行应用程序JAR包<br>
向Hadoop YARN集群管理器提交应用，需要把yarn-cluster作为主节点参数递给spark-submit。<br>

（2）在集群中运行pyspark<br>
也可以用pyspark连接到独立集群管理器上。<br><br>

**Reference:**<br>
1. [在集群上运行Spark应用程序(Python版)](http://dblab.xmu.edu.cn/blog/1699-2/)<br><br>

## 10. 使用spark计算《The man of property》中共出现过多少不重复的单词，以及出现次数最多的10个单词。 



**Reference:**<br>
1. [Spark案例：Python版统计单词个数](https://blog.csdn.net/howard2005/article/details/79331562)<br>
2. [Python3：Python+spark编程实战 总结](https://blog.csdn.net/proplume/article/details/79798289)<br><br>



## 11. 计算出movielen数据集中，平均评分最高的五个电影。


**Reference:**<br>
1. [基于Spark实现电影点评系统用户行为分析—RDD篇（一）](https://www.xuejiayuan.net/blog/34c3cddbce1241d88e476033796115ac)<br>
2. [ 第一篇：使用Spark探索经典数据集MovieLens](https://www.cnblogs.com/muchen/p/6881823.html)<br><br>


## 12. 计算出movielen中，每个用户最喜欢的前5部电影




## 13. 学会阅读Spark源码，整理Spark任务submit过程

**Reference:**<br>
1. [Spark源码阅读: Spark Submit任务提交](http://www.louisvv.com/archives/1340.html)<br>
2. [Spark 源码阅读（5）——Spark-submit任务提交流程](https://blog.csdn.net/qq_21989939/article/details/79598323)<br><br>









