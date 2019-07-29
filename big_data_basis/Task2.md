# 【Task 2】搭建Hadoop集群

## 1. 搭建HA的Hadoop集群并验证，3节点(1主2从)，理解HA/Federation,并截图记录搭建过程
以下所有内容在某个目录进行，比如在我的远程服务器上，我都在`/local/zy/tools/`目录进行
从百度云下载hadoop-2.7.3.tar.gz，再解压
```Bash
tar -zxvf hadoop-2.7.3.tar.gz
```
然后配置环境变量
```Bash
vi ~/.bashrc
# 加入
export HADOOP_HOME=/local/zy/tools/hadoop-2.7.3
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin
##
source ~/.bashrc
```
查看是否安装成功
```Bash
hadoop version
yarn version
```
创建hadoop文件目录
```Bash
mkdir data_hadoop
```
修改配置文件
```Bash
cd hadoop-2.7.3/etc/hadoop/
# 配置hdfs地址
vi core-site.xml
###
<configuration>
<property>
        <name>fs.default.name</name>
        <value>hdfs://node7:9000</value>
</property>
</configuration>
###
# 配置hdfs继承
###
<configuration>
<property>
        <name>dfs.name.dir</name> 
        <value>/local/zy/tools/data_hadoop/namenode</value>
</property>
<property>
        <name>dfs.data.dir</name>
        <value>/local/zy/tools/data_hadoop/datanode</value>
</property>
<property>
        <name>dfs.tmp.dir</name>
        <value>/local/zy/tools/data_hadoop/tmp</value>
</property>
<property>
        <name>dfs.replication</name>
        <value>2</value>
</property>
</configuration>
### 
# mapreduce相关配置
mv mapred-site.xml.template mapred-site.xml
vi mapred-site.xml
###
<configuration>
<property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
</property>
</configuration>
###
# 配置yarn
vi yarn-site.xml
###
<configuration>
<property>
        <name>yarn.resourcemanager.hostname</name>
        <value>node7</value>
</property>
<property>
        <name>yarn.resourcemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
</property>
</configuration>
###
# 配置slaves文件
vi slaves
###
node7
node8
node9
###
```
将hadoop和.bashrc文件夹复制到其他两个节点
```Bash
scp -r hadoop-2.7.3 zy@xxxx:/local/zy/tools
```
启动hdfs集群
```Bash
# 格式化namenode，在node7上执行
hdfs namenode -format
# 启动hdfs集群
start-dfs.sh
# jps验证
[zy@node7 tools]$ jps
1922 Jps
1653 SecondaryNameNode
1350 NameNode
[zy@node8 tools]$ jps
183451 DataNode
183564 Jps
[zy@node9 tools]$ jps
44029 DataNode
44142 Jps
```
小插曲：jps之后在node8和node9上没有发现datanode
```Bash
# 查看ｌｏｇ文件
less -S hadoop-2.7.3/logs/hadoop-zy-datanode-node9.log
发现其中的报错
###
2019-07-28 10:32:46,875 FATAL org.apache.hadoop.hdfs.server.datanode.DataNode: Exception in secureMain
java.net.BindException: Problem binding to [0.0.0.0:50010] java.net.BindException: Address already in use; For more details see:  http://wiki.apache.org/hadoop/BindException
###
# 提示地址被占用，所以就去寻找是什么占用的
[root@node9 tools]# netstat -nltp | grep 50010
tcp        0      0 0.0.0.0:50010           0.0.0.0:*               LISTEN      2115/java           
[root@node9 tools]# kill -9 2115
```
登录网页10.152.255.52:50070即可看到搭建的hadoop集群的网页界面<br>
测试hdfs的上传功能，先创建文件hello.txt，然后上传
```Bash
hdfs dfs -put hello.txt /hello.txt
```
然后就可以在网页的Utilities-Browse Directory里看到上传的文件了<br>
接下来启动yarn集群，并用jps在三个节点上检验，均没有问题
```Bash
[zy@node7 ~]$ start-yarn.sh 
starting yarn daemons
resourcemanager running as process 20722. Stop it first.
node9: starting nodemanager, logging to /local/zy/tools/hadoop-2.7.3/logs/yarn-zy-nodemanager-node9.out
node8: starting nodemanager, logging to /local/zy/tools/hadoop-2.7.3/logs/yarn-zy-nodemanager-node8.out
[zy@node7 ~]$ jps
19408 SecondaryNameNode
19121 NameNode
20722 ResourceManager
21022 Jps
[zy@node8 ~]$ jps
183451 DataNode
184312 Jps
184153 NodeManager
[zy@node9 ~]$ jps
44029 DataNode
45373 Jps
45214 NodeManager
```
登录网页10.152.255.52:8088即可看到yarn的管理界面<br><br>
对于HDFS的理解，参考https://blog.csdn.net/yinglish_/article/details/75269649<br>
对于HA/Federation的理解，参考https://blog.csdn.net/yinglish_/article/details/76785210<br>

## 2. 阅读Google三大论文，并总结
Google File System是一个可扩展的分布式文件系统，用于大型的、分布式的、对大量数据进行访问的应用。它运行于廉价的普通硬件上，提供容错功能。<br>
MapReduce是一种大数据的分布式计算方式，主要思想是将任务分解然后在多台处理能力较弱的计算节点中同时处理，然后将结果合并从而完成大数据处理。<br>
BigTable是 是建立在 GFS 和 MapReduce 之上的，系统采用Tablets的机制后，可以获得很好的负载均衡。<br>
参考资料：<br>
[Google三大理论（论文）](https://blog.csdn.net/w1573007/article/details/52966742)<br><br>


## 3. Hadoop的作用（解决了什么问题）/运行模式/基础组件及架构
### Hadoop的作用：
Hadoop 是一个开源软件框架，用于存储大量数据，并发处理/查询在具有多个商用硬件（即低成本硬件）节点的集群上的那些数据。
（参考：https://www.cnblogs.com/gala1021/p/8552850.html）<br>

### Hadoop的运行模式：
1）独立（本地）运行模式：无需任何守护进程，所有的程序都运行在同一个JVM上执行。在独立模式下调试MR程序非常高效方便。所以一般该模式主要是在学习或者开发阶段调试使用。<br>
2）伪分布式模式：  Hadoop守护进程运行在本地机器上，模拟一个小规模的集群，换句话说，可以配置一台机器的Hadoop集群,伪分布式是完全分布式的一个特例。<br>
3）完全分布式模式：Hadoop守护进程运行在一个集群上。<br>
（参考：https://blog.csdn.net/zane3/article/details/79829175）<br>

### Hadoop的基础组件及架构：
Hadoop是实现了分布式并行处理任务的系统框架，其核心组成是HDFS和MapReduce两个子系统，能够自动完成大任务计算和大数据储存的分割工作。<br>
HDFS系统是Hadoop的储存系统，能够实现创建文件、删除文件、移动文件等功能，操作的数据主要是要处理的原始数据以及计算过程中的中间数据，实现高吞吐量的数据读写。MapReduce系统是一个分布式计算框架，主要任务就是利用廉价的计算机对海量的数据进行分解处理。<br>

#### HDFS 架构
HDFS 是一个具有高度容错性的分布式文件系统， 适合部署在廉价的机器上。 HDFS 能提供高吞吐量的数据访问， 非常适合大规模数据集上的应用。HDFS 的架构如图所示， 总体上采用了 master/slave 架构， 主要由以下几个组件组成 ：Client、 NameNode、 Secondary NameNode 和 DataNode。 下面分别对这几个组件进行介绍：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/HDFS.jpg)<br>
**1) Client**<br>
Client（代表用户） 通过与 NameNode 和 DataNode 交互访问 HDFS 中的文件。 Client提供了一个类似 POSIX 的文件系统接口供用户调用。<br>
**2) NameNode**<br>
整个Hadoop 集群中只有一个 NameNode。 它是整个系统的“ 总管”， 负责管理 HDFS的目录树和相关的文件元数据信息。 这些信息是以“ fsimage”（ HDFS 元数据镜像文件）和“ editlog”（HDFS 文件改动日志）两个文件形式存放在本地磁盘，当 HDFS 重启时重新构造出来的。此外， NameNode 还负责监控各个 DataNode 的健康状态， 一旦发现某个DataNode 宕掉，则将该 DataNode 移出 HDFS 并重新备份其上面的数据。<br>
**3) Secondary NameNode**<br>
Secondary NameNode 最重要的任务并不是为 NameNode 元数据进行热备份， 而是定期合并 fsimage 和 edits 日志， 并传输给 NameNode。 这里需要注意的是，为了减小 NameNode压力， NameNode 自己并不会合并fsimage 和 edits， 并将文件存储到磁盘上， 而是交由Secondary NameNode 完成。<br>
**4) DataNode**<br>
一般而言， 每个 Slave 节点上安装一个 DataNode， 它负责实际的数据存储， 并将数据信息定期汇报给 NameNode。 DataNode 以固定大小的 block 为基本单位组织文件内容， 默认情况下 block 大小为 64MB。 当用户上传一个大的文件到 HDFS 上时， 该文件会被切分成若干个 block， 分别存储到不同的 DataNode ； 同时，为了保证数据可靠， 会将同一个block以流水线方式写到若干个（默认是 3，该参数可配置）不同的 DataNode 上。 这种文件切割后存储的过程是对用户透明的。<br>

#### MapReduce 架构
同 HDFS 一样，Hadoop MapReduce 也采用了 Master/Slave（M/S）架构，具体如图所示。它主要由以下几个组件组成：Client、JobTracker、TaskTracker 和 Task。 下面分别对这几个组件进行介绍。<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/MapReduce.jpg)<br>
**（1） Client**<br>
用户编写的 MapReduce 程序通过 Client 提交到 JobTracker 端； 同时， 用户可通过 Client 提供的一些接口查看作业运行状态。 在 Hadoop 内部用“作业”（Job） 表示 MapReduce 程序。 一个MapReduce 程序可对应若干个作业，而每个作业会被分解成若干个 Map/Reduce 任务（Task）。<br>
**（2） JobTracker**<br>
JobTracker 主要负责资源监控和作业调度。JobTracker监控所有TaskTracker与作业的健康状况，一旦发现失败情况后，其会将相应的任务转移到其他节点；同时JobTracker 会跟踪任务的执行进度、资源使用量等信息，并将这些信息告诉任务调度器，而调度器会在资源出现空闲时，选择合适的任务使用这些资源。在 Hadoop 中，任务调度器是一个可插拔的模块，用户可以根据自己的需要设计相应的调度器。<br>
**（3） TaskTracker**<br>
TaskTracker 会周期性地通过 Heartbeat 将本节点上资源的使用情况和任务的运行进度汇报给 JobTracker， 同时接收 JobTracker 发送过来的命令并执行相应的操作（如启动新任务、 杀死任务等）。TaskTracker 使用“slot” 等量划分本节点上的资源量。“slot” 代表计算资源（CPU、内存等）。一个Task 获取到一个slot 后才有机会运行，而Hadoop 调度器的作用就是将各个TaskTracker 上的空闲 slot 分配给 Task 使用。 slot 分为 Map slot 和 Reduce slot 两种，分别供 MapTask 和 Reduce Task 使用。 TaskTracker 通过 slot 数目（可配置参数）限定 Task 的并发度。<br>
**（4） Task**<br>
Task 分为 Map Task 和 Reduce Task 两种， 均由 TaskTracker 启动。 HDFS 以固定大小的 block 为基本单位存储数据， 而对于 MapReduce 而言， 其处理单位是 split。split 与 block 的对应关系如图所示。 split 是一个逻辑概念， 它只包含一些元数据信息， 比如数据起始位置、数据长度、数据所在节点等。它的划分方法完全由用户自己决定。 但需要注意的是，split 的多少决定了 Map Task 的数目 ，因为每个 split 会交由一个 Map Task 处理。<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/split_block.jpg)<br>
Map Task 执行过程如图所示。 由该图可知，Map Task 先将对应的 split 迭代解析成一个个 key/value 对，依次调用用户自定义的 map() 函数进行处理，最终将临时结果存放到本地磁盘上，其中临时数据被分成若干个 partition，每个 partition 将被一个Reduce Task 处理。<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/MapTask.jpg)<br>
Reduce Task 执行过程如图所示。该过程分为三个阶段①从远程节点上读取MapTask中间结果（称为“Shuffle 阶段”）；②按照key对key/value对进行排序（称为“ Sort 阶段”）；③依次读取<key, value list>，调用用户自定义的 reduce() 函数处理，并将最终结果存到 HDFS 上（称为“ Reduce 阶段”）。<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/MapReduce.jpg)<br>

## 4. 学会阅读HDFS源码，并自己阅读一段HDFS的源码(推荐HDFS上传/下载过程)
关于HDFS的文件上传，主要执行过程如下：<br>
１）FileSystem初始化，Client拿到NameNodeRpcServer代理对象，建立与NameNode的RPC通信<br>
２）调用FileSystem的create()方法，由于实现类为DistributedFileSystem,所有是调用该类中的create()方法<br>
３）DistributedFileSystem持有DFSClient的引用，继续调用DFSClient中的create()方法<br>
４）DFSOutputStream提供的静态newStreamForCreate()方法中调用NameNodeRpcServer服务端的create()方法并创建DFSOutputStream输出流对象返回<br>
５）通过hadoop提供的IOUtil工具类将输出流输出到本地<br><br>

下面我们来看下源码：<br><br>

首先初始化文件系统，建立与服务端的RPC通信<br>
```Java
HDFSDemo.java
OutputStream os = fs.create(new Path("/test.log"));
```
调用FileSystem的create()方法，由于FileSystem是一个抽象类，这里实际上是调用的该类的子类create()方法<br>
```Java
Copy1  //FileSystem.java
public abstract FSDataOutputStream create(Path f,
      FsPermission permission,
      boolean overwrite,
      int bufferSize,
      short replication,
      long blockSize,
      Progressable progress) throws IOException;
```
前面我们已经说过FileSystem.get()返回的是DistributedFileSystem对象，所以这里我们直接进入DistributedFileSystem：<br>
```Java
Copy 1   //DistributedFileSystem.java
@Override
  public FSDataOutputStream create(final Path f, final FsPermission permission,
    final EnumSet<CreateFlag> cflags, final int bufferSize,
    final short replication, final long blockSize, final Progressable progress,
    final ChecksumOpt checksumOpt) throws IOException {
    statistics.incrementWriteOps(1);
    Path absF = fixRelativePart(f);
    return new FileSystemLinkResolver<FSDataOutputStream>() {
      @Override
      public FSDataOutputStream doCall(final Path p)
          throws IOException, UnresolvedLinkException {
        final DFSOutputStream dfsos = dfs.create(getPathName(p), permission,
                cflags, replication, blockSize, progress, bufferSize,
                checksumOpt);
        //dfs为DistributedFileSystem所持有的DFSClient对象，这里调用DFSClient中的create()方法
        return dfs.createWrappedOutputStream(dfsos, statistics);
      }
      @Override
      public FSDataOutputStream next(final FileSystem fs, final Path p)
          throws IOException {
        return fs.create(p, permission, cflags, bufferSize,
            replication, blockSize, progress, checksumOpt);
      }
    }.resolve(this, absF);
  }
```
DFSClient的create()返回一个DFSOutputStream对象：<br>
```Java
Copy 1  //DFSClient.java
public DFSOutputStream create(String src, 
                             FsPermission permission,
                             EnumSet<CreateFlag> flag, 
                             boolean createParent,
                             short replication,
                             long blockSize,
                             Progressable progress,
                             int buffersize,
                             ChecksumOpt checksumOpt,
                             InetSocketAddress[] favoredNodes) throws IOException {
    checkOpen();
    if (permission == null) {
      permission = FsPermission.getFileDefault();
    }
    FsPermission masked = permission.applyUMask(dfsClientConf.uMask);
    if(LOG.isDebugEnabled()) {
      LOG.debug(src + ": masked=" + masked);
    }
    //调用DFSOutputStream的静态方法newStreamForCreate，返回输出流
    final DFSOutputStream result = DFSOutputStream.newStreamForCreate(this,
        src, masked, flag, createParent, replication, blockSize, progress,
        buffersize, dfsClientConf.createChecksum(checksumOpt),
        getFavoredNodesStr(favoredNodes));
    beginFileLease(result.getFileId(), result);
    return result;
  }
```
我们继续看下newStreamForCreate()中的业务逻辑：<br>
```Java
Copy 1 //DFSOutputStream.java
 static DFSOutputStream newStreamForCreate(DFSClient dfsClient, String src,
      FsPermission masked, EnumSet<CreateFlag> flag, boolean createParent,
      short replication, long blockSize, Progressable progress, int buffersize,
      DataChecksum checksum, String[] favoredNodes) throws IOException {
    TraceScope scope =
        dfsClient.getPathTraceScope("newStreamForCreate", src);
    try {
      HdfsFileStatus stat = null;
      boolean shouldRetry = true;
      int retryCount = CREATE_RETRY_COUNT;
      while (shouldRetry) {
        shouldRetry = false;
        try {
          //这里通过dfsClient的NameNode代理对象调用NameNodeRpcServer中实现的create()方法
          stat = dfsClient.namenode.create(src, masked, dfsClient.clientName,
              new EnumSetWritable<CreateFlag>(flag), createParent, replication,
              blockSize, SUPPORTED_CRYPTO_VERSIONS);
          break;
        } catch (RemoteException re) {
          IOException e = re.unwrapRemoteException(
              AccessControlException.class,
              DSQuotaExceededException.class,
              FileAlreadyExistsException.class,
              FileNotFoundException.class,
              ParentNotDirectoryException.class,
              NSQuotaExceededException.class,
              RetryStartFileException.class,
              SafeModeException.class,
              UnresolvedPathException.class,
              SnapshotAccessControlException.class,
              UnknownCryptoProtocolVersionException.class);
          if (e instanceof RetryStartFileException) {
            if (retryCount > 0) {
              shouldRetry = true;
              retryCount--;
            } else {
              throw new IOException("Too many retries because of encryption" +
                  " zone operations", e);
            }
          } else {
            throw e;
          }
        }
      }
      Preconditions.checkNotNull(stat, "HdfsFileStatus should not be null!");
     //new输出流对象
      final DFSOutputStream out = new DFSOutputStream(dfsClient, src, stat,
          flag, progress, checksum, favoredNodes);
      out.start();//调用内部类DataStreamer的start()方法，DataStreamer继承Thread，所以说这是一个线程，从NameNode中申请新的block信息；
　　　　　　　　　　　　　　　　同时前面我们介绍hdfs原理的时候提到的流水线作业（Pipeline）也是在这里实现，有兴趣的同学可以去研究下，这里就不带大家看了
      return out;
    } finally {
      scope.close();
    }
  }
```
到此，Client拿到了服务端的输出流对象，那么后面就容易了，都是一些简答的文件输出，输入流的操作（hadoop提供的IOUitl)。<br><br>

参考资料：<br>
[Hadoop之HDFS原理及文件上传下载源码分析（上）](https://www.cnblogs.com/qq503665965/p/6696675.html)<br>
[Hadoop之HDFS原理及文件上传下载源码分析（下）](https://www.cnblogs.com/qq503665965/p/6740992.html)<br><br>

## 5. Hadoop中各个组件的通信方式，RPC/Http等
RPC，即Remote Procedure Call，远程过程调用协议。<br>
概括的说，RPC采用客户机/服务器模式。请求程序就是一个客户机，而服务提供程序就是一个服务器。首先，客户机调用进程发送一个有进程参数的调用信息到服务进程，然后等待应答信息。在服务器端，进程保持睡眠状态直到调用信息的到达为止。当一个调用信息到达，服务器获得进程参数，计算结果，发送答复信息，然后等待下一个调用信息，最后，客户端调用进程接收答复信息，获得进程结果，然后调用执行继续进行。<br>
运行时,一次客户端对服务端的RPC调用,其内部操作大致有如下十步：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/RPC.jpeg)<br><br>
参考资料：<br>
[Hadoop基础知识小总结](https://www.cnblogs.com/pangzx/p/10296411.html)<br>
[Hadoop RPC机制-原理篇](https://blog.csdn.net/u010010428/article/details/51345693)<br>
<br><br>


## 6. 学会写WordCount（Java/Python-Hadoop Streaming），理解分布式/单机运行模式的区别
编写map函数 wordcount_mapper.py
```Python
#!/usr/bin/env python   
#This mapper code will input a line of text and output <word, 1>

import sys            

for line in sys.stdin:  
    line = line.strip()  
    keys = line.split() 
    for key in keys:    
        value = 1        
        #the {} is replaced by 0th,1st items in format list
        print('{0}\t{1}'.format(key, value) ) 
                       
```
reduce函数 word count_reducer.py
```Python
#!/usr/bin/env python

#This reducer code will input a line of text and 
#    output <word, total-count>
import sys

last_key      = None              
running_total = 0

# 使用循环读取输入并计数
for input_line in sys.stdin:
    input_line = input_line.strip()
    this_key, value = input_line.split("\t", 1) 
    value = int(value)           
 
    if last_key == this_key:     
        running_total += value   # add value to running total

    else:
        if last_key:          
            print( "{0}\t{1}".format(last_key, running_total) )
                               
        running_total = value    #reset values
        last_key = this_key

if last_key == this_key:
    print( "{0}\t{1}".format(last_key, running_total)) 

```
streaming 使用绝对地址，output 不能是已经存在的目录 mapper 和reducer使用绝对地址，然后运行命令，
```Bash
hadoop jar /Download/hadoop-streaming-2.7.3.jar 
-input /hello \
-output /output 
-mapper /usr/local/yarn/hadoop-2.7.3/wordcount/wordcount_mapper.py 
-reducer /usr/local/yarn/hadoop-2.7.3/wordcount/wordcount_reducer.py
```
然后查看/output就可以看到结果。<br><br>

参考资料：<br>
[使用Hadoop Streaming运行Python版Wordcount](https://www.jianshu.com/p/e3fba578d1a8)<br><br>

## 7. 理解MapReduce的执行过程
假设用户编写了一个 MapReduce 程序，并将其打包成 xxx.jar 文件，然后使用 以下命令提交作业：<br>
```Bash
$HADOOP_HOME/bin/hadoop jar xxx.jar \
-D mapred.job.name="xxx" \
-D mapred.map.tasks=3 \
-D mapred.reduce.tasks=2 \
-D input=/test/input \
-D output=/test/output
```
则该作业的运行过程如图所示：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/MapReduce_execute.jpg)<br>
这个过程分为以下 5 个步骤：<br>
步骤 1：作业提交与初始化。 用户提交作业后， 首先由 JobClient 实例将作业相关信息， 比如将程序 jar 包、作业配置文件、 分片元信息文件等上传到分布式文件系统（ 一般为HDFS）上，其中，分片元信息文件记录了每个输入分片的逻辑位置信息。 然后 JobClient通过 RPC 通知 JobTracker。 JobTracker 收到新作业提交请求后， 由 作业调度模块对作业进行初始化：为作业创建一个 JobInProgress 对象以跟踪作业运行状况， 而 JobInProgress 则会为每个Task 创建一个 TaskInProgress 对象以跟踪每个任务的运行状态， TaskInProgress 可能需要管理多个“ Task 运行尝试”（ 称为“ Task Attempt”）。<br>
步骤 2：任务调度与监控。 前面提到，任务调度和监控的功能均由 JobTracker 完成。TaskTracker 周期性地通过 Heartbeat 向 JobTracker 汇报本节点的资源使用 情况， 一旦出 现空闲资源， JobTracker 会按照一定的策略选择一个合适的任务使用该空闲资源， 这由任务调度器完成。 任务调度器是一个可插拔的独立模块， 且为双层架构， 即首先选择作业， 然后从该作业中选择任务， 其中，选择任务时需要重点考虑数据本地性。 此外，JobTracker 跟踪作业的整个运行过程，并为作业的成功运行提供全方位的保障。 首先， 当 TaskTracker 或者Task 失败时， 转移计算任务 ； 其次， 当某个 Task 执行进度远落后于同一作业的其他 Task 时，为之启动一个相同 Task， 并选取计算快的 Task 结果作为最终结果。<br>
步骤 3：任务运行环境准备。 运行环境准备包括 JVM 启动和资源隔 离， 均由TaskTracker 实现。 TaskTracker 为每个 Task 启动一个独立的 JVM 以避免不同 Task 在运行过程中相互影响 ； 同时，TaskTracker 使用了操作系统进程实现资源隔离以防止 Task 滥用资源。 <br>
步骤 4 ：任务执行。 TaskTracker 为 Task 准备好运行环境后， 便会启动 Task。 在运行过程中， 每个 Task 的最新进度首先由 Task 通过 RPC 汇报给 TaskTracker， 再由 TaskTracker汇报给 JobTracker。 <br>
步骤 5 作业完成。 待所有 Task 执行完毕后， 整个作业执行成功。<br><br>


参考资料：<br>
[Hadoop 基本架构](https://blog.csdn.net/weixin_33728708/article/details/90662604)<br><br>

## 8. Yarn在Hadoop中的作用
hadoop1.x是有一些问题的，其中重要的问题是jobTracker的问题，例如单点故障，任务过重，其中除了JobTracker同时还有一个TaskTracker。为了更好的理解yarn的作用，我们就需要跟hadoop1.x比较。<br>
在hadoop２.x中，我们看到JobTracker的功能被分散到各个进程中包括ResourceManager和NodeManager：比如监控功能，分给了NodeManager，和Application Master。ResourceManager里面又分为了两个组件：调度器及应用程序管理器。也就是说Yarn重构后，JobTracker的功能，被分散到了各个进程中。同时由于这些进程可以被单独部署所以这样就大大减轻了单点故障及压力。同时我们还看到Yarn使用了Container，而hadoop1.x中使用了slot。slot存在的缺点比如只能map或则reduce用。Container则不存在这个问题。这也是Yarn的进步。<br>

参考资料：<br>
[hadoop为什么会有yarn](https://www.jianshu.com/p/883a4f459bed)<br><br>





