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
node8
node9
###
```
