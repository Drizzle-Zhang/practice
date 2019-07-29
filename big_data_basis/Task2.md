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
对于HA/Federation的理解，参考https://blog.csdn.net/yinglish_/article/details/76785210

## 2. 阅读Google三大论文，并总结


## 3. Hadoop的作用（解决了什么问题）/运行模式/基础组件及架构
Hadoop的作用：Hadoop 是一个开源软件框架，用于存储大量数据，并发处理/查询在具有多个商用硬件（即低成本硬件）节点的集群上的那些数据。
（参考：https://www.cnblogs.com/gala1021/p/8552850.html）<br><br>
Hadoop的运行模式：


