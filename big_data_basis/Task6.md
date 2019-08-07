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
安装完成后，启动MySQL服务
```
service mysqld start
```
重置密码
```
(base) [root@node7 src]# grep "password" /var/log/mysqld.log    
2019-08-07T01:06:41.847256Z 1 [Note] A temporary password is generated for root@localhost: 9r2,eSv)/q_f   # 默认密码
(base) [root@node7 src]# mysql -u root -p
Enter password:  # 输入密码进入

mysql> alter user 'root'@'localhost' identified by 'zy123456' # 修改密码
    ->   flush privileges        # 刷新权限

```
