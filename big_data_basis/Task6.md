# 【Task6】Hive原理及其使用

## 1. 安装MySQL、Hive

### 安装Hive
先解压缩
```
[zy@node7 tools]$ tar -vxzf apache-hive-2.1.1-bin.tar.gz
```
配置环境变量
```
export HIVE_HOME=/local/zy/tools/apache-hive-2.1.1-bin
export PATH=$PATH:$HIVE_HOME/bin
```

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
重启MySQL，并用新密码登录
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

mysql> 

```

**Reference:**<br>
1. [CentOS 7 下使用yum安装MySQL5.7.20 最简单 图文详解](https://blog.csdn.net/z13615480737/article/details/78906598)<br>
2. [centos7安装mysql5.7](https://blog.51cto.com/13941177/2176400)<br><br>









