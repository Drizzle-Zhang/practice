# 【Task6】Hive原理及其使用

## 1. 安装MySQL、Hive

### 安装MySQL
理论上应该先卸载之前的MySQL，但是我的机器上之前没有安装MySQL，因此略过<br>
先下载MySQL的repo源,然后安装MySQL
```
(base) [root@node7 zy]# cd /usr/local/src/
(base) [root@node7 src]# ls
(base) [root@node7 src]# cd /usr/local/src/^C
(base) [root@node7 src]# wget http://repo.mysql.com/mysql57-community-release-el7-8.noarch.rpm
(base) [root@node7 src]# rpm -ivh mysql57-community-release-el7-8.noarch.rpm 
(base) [root@node7 src]# yum -y install mysql-server 
```
安装完成后，启动MySQL服务，并查看MySQL状态
```
(base) [root@node7 src]# systemctl start mysqld.service
(base) [root@node7 src]# systemctl status mysqld.service
● mysqld.service - MySQL Server
   Loaded: loaded (/usr/lib/systemd/system/mysqld.service; enabled; vendor preset: disabled)
   Active: active (running) since Wed 2019-08-07 09:06:45 CST; 18min ago
     Docs: man:mysqld(8)
           http://dev.mysql.com/doc/refman/en/using-systemd.html
 Main PID: 40200 (mysqld)
   CGroup: /system.slice/mysqld.service
           └─40200 /usr/sbin/mysqld --daemonize --pid-file=/var/run/mysqld/...

Aug 07 09:06:38 node7 systemd[1]: Starting MySQL Server...
Aug 07 09:06:45 node7 systemd[1]: Started MySQL Server.
```
重置密码
```
(base) [root@node7 src]# grep "password" /var/log/mysqld.log    
2019-08-07T01:06:41.847256Z 1 [Note] A temporary password is generated for root@localhost: 9r2,eSv)/q_f   # 默认密码
(base) [root@node7 src]# mysql -u root -p
Enter password:  # 输入密码进入
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 6
Server version: 5.7.27

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> set global validate_password_policy=0;
Query OK, 0 rows affected (0.00 sec)

mysql> set global validate_password_length=1;
Query OK, 0 rows affected (0.00 sec)

mysql> set global validate_password_mixed_case_count=2;
Query OK, 0 rows affected (0.00 sec)

mysql> alter user 'root'@'localhost' identified by 'zy123456';
Query OK, 0 rows affected (0.00 sec)     # 修改密码

mysql> flush privileges;
Query OK, 0 rows affected (0.00 sec)     # 刷新权限

```
重启MySQL，并用新密码登录，创建一个普通用户
```
(base) [root@node7 src]# service mysqld restart
Redirecting to /bin/systemctl restart  mysqld.service
(base) [root@node7 src]# mysql -u root -p
Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 2
Server version: 5.7.27 MySQL Community Server (GPL)

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> set global validate_password_policy=0;
Query OK, 0 rows affected (0.00 sec)

mysql> set global validate_password_length=1;
Query OK, 0 rows affected (0.00 sec)

mysql> set global validate_password_mixed_case_count=2;
Query OK, 0 rows affected (0.00 sec)

mysql> create user 'zy'@'localhost' identified by '123456';
Query OK, 0 rows affected (0.00 sec)

mysql> flush privileges;
Query OK, 0 rows affected (0.00 sec) 

```
普通用户可以登录了
```
[zy@node7 ~]$ mysql -u zy -p
Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 4
Server version: 5.7.27 MySQL Community Server (GPL)

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> 

```

### 安装并配置Hive
先解压缩
```
[zy@node7 tools]$ tar -vxzf apache-hive-2.1.1-bin.tar.gz
```
配置环境变量
```
export HIVE_HOME=/local/zy/tools/apache-hive-2.1.1-bin
export PATH=$PATH:$HIVE_HOME/bin
```
<br>

**Reference:**<br>
1. [CentOS 7 下使用yum安装MySQL5.7.20 最简单 图文详解](https://blog.csdn.net/z13615480737/article/details/78906598)<br>
2. [centos7安装mysql5.7](https://blog.51cto.com/13941177/2176400)<br>
<br>



## 2. 采用MySQL作为hive元数据库
创建hive-site.xml 文件
```
[zy@node7 tools]$ cd apache-hive-2.1.1-bin/
[zy@node7 apache-hive-2.1.1-bin]$ cd conf/
[zy@node7 conf]$ cp hive-default.xml.template hive-site.xml 
```
创建HDFS文件夹<br>
先查看hive-site.xml 中HDFS相关配置
```
  <property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse</value>
    <description>location of default database for the warehouse</description>
  </property>

  <property>
    <name>hive.exec.scratchdir</name>
    <value>/tmp/hive</value>
    <description>HDFS root scratch dir for Hive jobs which gets created with write all (733) permission. For each connecting user, an HDFS scratch dir: ${hive.exec.scratchdir}/&lt;username&gt; is created, with ${hive.scratch.dir.permission}.</description>
  </property>
```
因此我们需要现在HDFS中创建对应的目录，并修改相应的权限
```
[zy@node7 ~]$ hadoop fs -mkdir -p /user/hive/warehouse
[zy@node7 ~]$ hadoop fs -mkdir -p /tmp/hive
[zy@node7 ~]$ hadoop fs -chmod -R 777 /user/hive/warehouse
[zy@node7 ~]$ hadoop fs -chmod -R 777 /tmp/hive
[zy@node7 ~]$ hadoop fs -ls /
Found 4 items
-rw-r--r--   2 zy supergroup          7 2019-07-28 15:27 /hello.txt
drwxr-xr-x   - zy supergroup          0 2019-07-31 05:43 /test
drwx-wx-wx   - zy supergroup          0 2019-08-05 09:48 /tmp
drwxr-xr-x   - zy supergroup          0 2019-08-07 09:50 /user
```
Hive相关配置<br>
将 hive-site.xml 中的{system:java.io.tmpdir}改为hive的本地临时目录，将{system:user.name}改为用户名。<br>
如果该目录不存在，需要先创建该目录。
```
[zy@node7 apache-hive-2.1.1-bin]$ mkdir tmp
[zy@node7 apache-hive-2.1.1-bin]$ cd tmp/
[zy@node7 tmp]$ pwd
/local/zy/tools/apache-hive-2.1.1-bin/tmp
```
修改hive-site.xml 中的配置如下：
```
  <property>
    <name>hive.exec.local.scratchdir</name>
    <value>/local/zy/tools/apache-hive-2.1.1-bin/tmp/zy</value>
    <description>Local scratch space for Hive jobs</description>
  </property>
  <property>
    <name>hive.downloaded.resources.dir</name>
    <value>/local/zy/tools/apache-hive-2.1.1-bin/tmp/${hive.session.id}_resources</value>
    <description>Temporary local directory for added resources in the remote file system.</description>
  </property>

  <property>
    <name>hive.server2.logging.operation.log.location</name>
    <value>/local/zy/tools/apache-hive-2.1.1-bin/tmp/zy/operation_logs</value>
    <description>Top level directory where operation logs are stored if logging functionality is enabled</description>
  </property>
  
  <property>
    <name>hive.querylog.location</name>
    <value>/local/zy/tools/apache-hive-2.1.1-bin/tmp/zy</value>
    <description>Location of Hive run time structured log file</description>
  </property>

```
数据库相关配置<br>
同样修改 hive-site.xml 中的以下几项，注意驱动版本的问题，否则后面会报错：
```
# 数据库jdbc地址，value标签内修改为主机ip地址
  <property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://node7:3306/hive?createDatabaseIfNotExist=true&amp;characterEncoding=UTF-8</value>
    <description>
      JDBC connect string for a JDBC metastore.
      To use SSL to encrypt/authenticate the connection, provide database-specific SSL flag in the connection URL.
      For example, jdbc:postgresql://myhost/db?ssl=true for postgres database.
    </description>
  </property>

# 数据库的驱动类名称
# 新版本8.0版本的驱动为com.mysql.cj.jdbc.Driver
# 旧版本5.x版本的驱动为com.mysql.jdbc.Driver
  <property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.cj.jdbc.Driver</value>
  </property>
  
# 数据库用户名
  <property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>root</value>
  </property>
 
# 数据库密码
   <property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>zy123456</value> #修改为你自己的mysql密码
  </property>

  <property>
    <name>hive.metastore.schema.verification</name>
    <value>false</value>
  </property>

```
复制并更名hive-log4j2.properties.template为 hive-log4j2.properties文件：
```
[zy@node7 conf]$ cp hive-log4j2.properties.template hive-log4j2.properties
[zy@node7 conf]$ vi hive-log4j2.properties
```
修改如下内容：
```
property.hive.log.dir = /local/zy/tools/apache-hive-2.1.1-bin/tmp/zy
```
复制并更名hive-env.sh.template为 hive-env.sh文件：
```
[zy@node7 ~]$ cd tools/apache-hive-2.1.1-bin/conf/
[zy@node7 conf]$ cp hive-e
hive-env.sh.template                  hive-exec-log4j2.properties.template
[zy@node7 conf]$ cp hive-env.sh.template hive-env.sh
[zy@node7 conf]$ vi hive-env.sh
```
修改如下内容
```
# Set HADOOP_HOME to point to a specific hadoop install directory
# HADOOP_HOME=${bin}/../../hadoop
export HADOOP_HOME=/local/zy/tools/hadoop-2.7.3     # hadoop安装目录

# Hive Configuration Directory can be controlled by:
export HIVE_CONF_DIR=/local/zy/tools/apache-hive-2.1.1-bin/conf       # hive配置文件目录

# Folder containing extra ibraries required for hive compilation/execution can be controlled by:
export HIVE_AUX_JARS_PATH=/local/zy/tools/apache-hive-2.1.1-bin/lib         # hive依赖jar包目录

```
接下来启动Hive<br>
进入MySQL，创建Hive数据库
```
[zy@node7 ~]$ mysql -u root -p
Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 6
Server version: 5.7.27 MySQL Community Server (GPL)

Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql> create database hive;
Query OK, 1 row affected (0.00 sec)

```
进入官网下载数据库驱动，在System选择栏里选Platform Independent，选择后缀tar.gz的包进行下载并解压，然后将里面的jar复制到$HIVE_HOME/lib，再进行初始化
```
[zy@node7 download]$ cp mysql-connector-java-8.0.17/*.jar ~/tools/apache-hive-2.1.1-bin/lib/
[zy@node7 ~]$ schematool -dbType mysql -initSchema
```
会出现如下报错：
```
which: no hbase in (/usr/local/matlab/2014a/bin:/opt/scm/bin:/opt/apps/python/3.6.1/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ibutils/bin:/local/zy/tools/hadoop-2.7.3/bin:/local/zy/tools/hadoop-2.7.3/sbin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.151-1.b12.el7_4.x86_64/jre/bin:/local/zy/tools/spark-2.1.1-bin-hadoop2.7/bin:/local/zy/tools/apache-hive-2.1.1-bin/bin:/local/zy/tools/anaconda3/bin:/local/zy/.local/bin:/local/zy/bin)
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/local/zy/tools/apache-hive-2.1.1-bin/lib/log4j-slf4j-impl-2.4.1.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/local/zy/tools/hadoop-2.7.3/share/hadoop/common/lib/slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Metastore connection URL:	 jdbc:mysql://node7:3306/hive?createDatabaseIfNotExist=true&characterEncoding=UTF-8
Metastore Connection Driver :	 com.mysql.cj.jdbc.Driver
Metastore connection User:	 zy
org.apache.hadoop.hive.metastore.HiveMetaException: Failed to get schema version.
Underlying cause: java.sql.SQLException : null,  message from server: "Host 'node7' is not allowed to connect to this MySQL server"
SQL Error code: 1130
Use --verbose for detailed stacktrace.
*** schemaTool failed ***
```
服务器不允许远程连接，我们进入mysql进行以下修改：
```
mysql> use mysql
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
mysql> update user set host = '%' where user='root';
Query OK, 1 row affected (0.00 sec)
Rows matched: 1  Changed: 1  Warnings: 0

mysql> flush privileges;
Query OK, 0 rows affected (0.00 sec) 

```
之后再次进行初始化，初始化成功
```
[zy@node7 ~]$ schematool -dbType mysql -initSchema
which: no hbase in (/usr/local/matlab/2014a/bin:/opt/scm/bin:/opt/apps/python/3.6.1/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ibutils/bin:/local/zy/tools/hadoop-2.7.3/bin:/local/zy/tools/hadoop-2.7.3/sbin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.151-1.b12.el7_4.x86_64/jre/bin:/local/zy/tools/spark-2.1.1-bin-hadoop2.7/bin:/local/zy/tools/apache-hive-2.1.1-bin/bin:/local/zy/tools/anaconda3/bin:/local/zy/.local/bin:/local/zy/bin)
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/local/zy/tools/apache-hive-2.1.1-bin/lib/log4j-slf4j-impl-2.4.1.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/local/zy/tools/hadoop-2.7.3/share/hadoop/common/lib/slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.apache.logging.slf4j.Log4jLoggerFactory]
Metastore connection URL:	 jdbc:mysql://node7:3306/hive?createDatabaseIfNotExist=true&characterEncoding=UTF-8
Metastore Connection Driver :	 com.mysql.cj.jdbc.Driver
Metastore connection User:	 root
Starting metastore schema initialization to 2.1.0
Initialization script hive-schema-2.1.0.mysql.sql
Initialization script completed
schemaTool completed
```


**Reference:**<br>
1. [Linux CentOS7 安装Hive3.1.1](https://blog.csdn.net/qq_39680564/article/details/89714184#42__44)<br>
2. [CentOs7 Hadoop-3.0.3 搭建Hive-2.3.5](https://blog.csdn.net/zhouzhiwengang/article/details/94576029)<br>
3. [CentOS7+ Hadoop3.2.0+MySQL5.7安装配置Hive3.1.1](https://blog.csdn.net/qq_39315740/article/details/98626518)<br>
<br>


## 3. Hive与传统RDBMS的区别
1、hive存储的数据量比较大，适合海量数据，适合存储轨迹类历史数据，适合用来做离线分析、数据挖掘运算，事务性较差，实时性较差；<br>
 rdbms一般数据量相对来说不会太大，适合事务性计算，实时性较好，更加接近上层业务<br>

2、hive的计算引擎是hadoop的mapreduce，存储是hadoop的hdfs文件系统，<br>
 rdbms的引擎由数据库自己设计实现例如mysql的innoDB，存储用的是数据库服务器本地的文件系统<br>

3、hive由于基于hadoop所以存储和计算的扩展能力都很好，<br>
 rdbms在这方面比较弱，比如orcale的分表和扩容就很头疼<br>

4、hive表格没有主键、没有索引、不支持对具体某一行的操作，适合对批量数据的操作，不支持对数据的update操作，更新的话一般是先删除表然后重新落数据<br>
 rdbms事务性强，有主键、索引，支持对具体某一行的增删改查等操作<br>

5、hive的SQL为HQL，与标准的RDBMS的SQL存在有不少的区别，相对来说功能有限<br>
rdbms的SQL为标准SQL，功能较为强大。<br>

6、Hive在加载数据时候和rdbms关系数据库不同，hive在加载数据时候不会对数据进行检查，也不会更改被加载的数据文件，
而检查数据格式的操作是在查询操作时候执行，这种模式叫“读时模式”。在实际应用中，写时模式在加载数据时候会对列进行
索引，对数据进行压缩，因此加载数据的速度很慢，但是当数据加载好了，我们去查询数据的时候，速度很快。但是当我们的
数据是非结构化，存储模式也是未知时候，关系数据操作这种场景就麻烦多了，这时候hive就会发挥它的优势。<br>

 rdbms里，表的加载模式是在数据加载时候强制确定的（表的加载模式是指数据库存储数据的文件格式），如果加载数据
时候发现加载的数据不符合模式，关系数据库则会拒绝加载数据，这个就叫“写时模式”，写时模式会在数据加载时候对数据模
式进行检查校验的操作。<br><br>

总的来说，Hive是数据仓库适合存储历史的海量的数据，适合做批量和海量复杂运算，事务性差，运算时间长。<br>
RDBMS是数据库，存储数据量偏小一些，事务性强，适合做OLTP和OLAP业务，运算时间短。<br><br>

对于Hive和传统RMBMS的区别，可以简单总结为下面的表格：<br>
![](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/supp_Task6/Hive_RDBMS.png)<br><br>

**Reference:**<br>
1. [HIVE与RDBMS的区别](https://blog.csdn.net/zx8167107/article/details/79114620)<br>
2. [Hive 与 RDBMS的区别](https://blog.csdn.net/qq_43688472/article/details/86317428)<br>
3. [Hive和传统数据库区别总结](https://blog.csdn.net/peter_changyb/article/details/81219460)<br>
<br>



