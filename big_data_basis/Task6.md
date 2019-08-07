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

```



**Reference:**<br>
1. [CentOS 7 下使用yum安装MySQL5.7.20 最简单 图文详解](https://blog.csdn.net/z13615480737/article/details/78906598)<br>
2. [centos7安装mysql5.7](https://blog.51cto.com/13941177/2176400)<br><br>









