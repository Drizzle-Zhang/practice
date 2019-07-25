# 【Task1】创建虚拟机+熟悉linux(2day)

1. [创建三台虚拟机](https://mp.weixin.qq.com/s/WkjX8qz7nYvuX4k9vaCdZQ)
2. 在本机使用Xshell连接虚拟机
3. [CentOS7配置阿里云yum源和EPEL源](https://www.cnblogs.com/jimboi/p/8437788.html)
4. 安装jdk
5. 熟悉linux 常用命令
6. 熟悉，shell 变量/循环/条件判断/函数等

shell小练习1：
编写函数，实现将1-100追加到output.txt中，其中若模10等于0，则再追加输出一次。即10，20...100在这个文件中会出现两次。

注意：
* 电脑系统需要64位(4g+)
* 三台虚拟机的运行内存不能超过电脑的运行内存
* 三台虚拟机ip不能一样，否则会有冲突、



参考资料：
 1. [安装ifconfig](https://jingyan.baidu.com/article/363872ec26bd0f6e4aa16f59.html)
 2. [bash: wget: command not found的两种解决方法](https://www.cnblogs.com/areyouready/p/8909665.html)
 3. linux系统下载ssh服务
 4. [关闭windows的防火墙！如果不关闭防火墙的话，可能和虚拟机直接无法ping通！](https://www.linuxidc.com/Linux/2017-11/148427.htm)
 5. 大数据软件 ：[链接](https://pan.baidu.com/s/17fEq3IPVoeE29cWCrSpO8Q) 提取码：finf

## 1. 创建三台虚拟机
先安装virtualbox(Ubuntu18.04)
```Bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install virtualbox
```
按照[题目中链接](https://mp.weixin.qq.com/s/WkjX8qz7nYvuX4k9vaCdZQ)配置虚拟机,三台虚拟机分别给1GB内存，10GB硬盘<br>
注：<br>
1）网卡名称的选项和win系统不太一样，先搁置看后面有什么影响<br>
2）启动虚拟机时遇到不能打开虚拟机的问题，在群里询问之后发现是BIOS里面的Intel VT选项没有Enable，修改BIOS设置之后问题解决<br>
3）因为电脑连的无线校园网，所以IP是动态的，在安装步骤里先不设置IPv4地址<br>

## 2. 在本机连接虚拟机
先对网络进行设置，打开网卡文件<br>
```Bash
sudo vi /etc/sysconfig/network-scripts/ifcfg-enp0s3
```
在文件后面加上以下网络信息：
```Bash
ONBOOT=yes 
BOOTPROTO=static 
IPADDR=202.120.234.143
NETMASK=255.255.255.0 
GATEWAY=10.157.56.1
DNS1=192.168.1.1
DNS2=8.8.8.8
```
然后重启网络服务,并开启sshd服务
```Bash
sudo systemctl restart network
sudo service sshd start
```
尝试ping本机IP
```Bash
ping 202.120.234.142
```
结果无法ping通，设置网络无法成功`哭晕`跳过直接进入第三步吧<br>

因为虚拟环境搭建失败，以下步骤在现成的CentOS上进行：

## 3. CentOS7配置阿里云yum源和EPEL源
该部分按照[教程](https://www.cnblogs.com/jimboi/p/8437788.html)进行<br>
首先，备份系统里原有的源
```Bash
[zy@node9 ~]$ cd /etc/yum.repos.d/
[zy@node9 yum.repos.d]$ su  # root
Password: 
[root@node9 yum.repos.d]# mkdir repo_bak
[root@node9 yum.repos.d]# ls
CentOS-Base.repo       CentOS-fasttrack.repo  CentOS-Vault.repo  repo_bak
CentOS-CR.repo         CentOS-Media.repo      epel.repo
CentOS-Debuginfo.repo  CentOS-Sources.repo    epel-testing.repo
[root@node9 yum.repos.d]# mv *.repo repo_bak/
[root@node9 yum.repos.d]# ls
repo_bak
```
下载新的CentOS-Base.repo 到/etc/yum.repos.d/
```Bash
[root@node9 yum.repos.d]# wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
--2019-07-25 02:42:04--  http://mirrors.aliyun.com/repo/Centos-7.repo
Resolving mirrors.aliyun.com (mirrors.aliyun.com)... 222.22.29.99, 222.22.29.100, 222.22.29.101, ...
Connecting to mirrors.aliyun.com (mirrors.aliyun.com)|222.22.29.99|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 139 [text/html]
Saving to: ‘/etc/yum.repos.d/CentOS-Base.repo’

100%[=======================================>] 139         --.-K/s   in 0s      

2019-07-25 02:42:04 (25.3 MB/s) - ‘/etc/yum.repos.d/CentOS-Base.repo’ saved [139/139]

[root@node9 yum.repos.d]# ls
CentOS-Base.repo  repo_bak


[root@node9 yum.repos.d]# yum clean all     # 清除缓存
[root@node9 yum.repos.d]# yum makecache     # 生成新缓存

```
安装EPEL（Extra Packages for Enterprise Linux ）源
```Bash
[root@node9 yum.repos.d]# yum install -y epel-release
Loaded plugins: fastestmirror, langpacks
Repodata is over 2 weeks old. Install yum-cron? Or run: yum makecache fast
base                                                      | 3.6 kB  00:00:00     
epel/x86_64/metalink                                      | 7.6 kB  00:00:00     
epel                                                      | 5.3 kB  00:00:00     
extras                                                    | 3.4 kB  00:00:00     
updates                                                   | 3.4 kB  00:00:00     
(1/6): base/7/x86_64/group_gz                             | 166 kB  00:00:00     
(2/6): base/7/x86_64/primary_db                           | 6.0 MB  00:00:01     
(3/6): epel/x86_64/updateinfo                             | 996 kB  00:00:02     
(4/6): extras/7/x86_64/primary_db                         | 205 kB  00:00:00     
(5/6): updates/7/x86_64/primary_db                        | 6.5 MB  00:00:01     
(6/6): epel/x86_64/primary_db                             | 6.8 MB  00:00:06     
Determining fastest mirrors
 * base: ftp.sjtu.edu.cn
 * epel: mirror01.idc.hinet.net
 * extras: ftp.sjtu.edu.cn
 * updates: ftp.sjtu.edu.cn
Resolving Dependencies
--> Running transaction check
---> Package epel-release.noarch 0:7-9 will be updated
---> Package epel-release.noarch 0:7-11 will be an update
--> Finished Dependency Resolution

Dependencies Resolved

=================================================================================
 Package                 Arch              Version         Repository       Size
=================================================================================
Updating:
 epel-release            noarch            7-11            epel             15 k

Transaction Summary
=================================================================================
Upgrade  1 Package

Total download size: 15 k
Downloading packages:
epel/x86_64/prestodelta                                   | 1.7 kB  00:00:00     
epel-release-7-11.noarch.rpm                              |  15 kB  00:00:00     
Running transaction check
Running transaction test
Transaction test succeeded
Running transaction
  Updating   : epel-release-7-11.noarch                                      1/2 
  Cleanup    : epel-release-7-9.noarch                                       2/2 
  Verifying  : epel-release-7-11.noarch                                      1/2 
  Verifying  : epel-release-7-9.noarch                                       2/2 

Updated:
  epel-release.noarch 0:7-11                                                     

Complete!

```

## 4. 安装jdk
参考链接：https://www.cnblogs.com/116970u/p/10400436.html<br>
查看当前java版本
```Bash
[root@node9 yum.repos.d]# java -version
openjdk version "1.8.0_181"
OpenJDK Runtime Environment (build 1.8.0_181-b13)
OpenJDK 64-Bit Server VM (build 25.181-b13, mixed mode)
```
卸载系统自带的OpenJDK以及相关的java文件
```Bash
[root@node9 yum.repos.d]# rpm -qa | grep java
R-java-3.5.0-1.el7.x86_64
java-1.8.0-openjdk-headless-1.8.0.181-3.b13.el7_5.x86_64
java-1.7.0-openjdk-headless-1.7.0.161-2.6.12.0.el7_4.x86_64
javapackages-tools-3.4.1-11.el7.noarch
java-1.8.0-openjdk-devel-1.8.0.181-3.b13.el7_5.x86_64
python-javapackages-3.4.1-11.el7.noarch
R-java-devel-3.5.0-1.el7.x86_64
java-1.7.0-openjdk-1.7.0.161-2.6.12.0.el7_4.x86_64
java-1.8.0-openjdk-1.8.0.181-3.b13.el7_5.x86_64
tzdata-java-2017c-1.el7.noarch
[root@node9 yum.repos.d]# rpm -e --nodeps java-1.8.0-openjdk-headless-1.8.0.181-3.b13.el7_5.x86_64
[root@node9 yum.repos.d]# rpm -e --nodeps java-1.7.0-openjdk-headless-1.7.0.161-2.6.12.0.el7_4.x86_64
[root@node9 yum.repos.d]# rpm -e --nodeps java-1.7.0-openjdk-1.7.0.161-2.6.12.0.el7_4.x86_64
[root@node9 yum.repos.d]# rpm -e --nodeps java-1.8.0-openjdk-1.8.0.181-3.b13.el7_5.x86_64
[root@node9 yum.repos.d]# java -version
bash: /usr/bin/java: No such file or directory
```
下载最新稳定JDK
```Bash
wget https://download.oracle.com/otn-pub/java/jdk/12.0.2+10/e482c34c86bd4bf8b56c0b35558996b9/jdk-12.0.2_linux-x64_bin.tar.gz
mkdir /usr/local/java
cp /local/zy/tools/jdk-12.0.2_linux-x64_bin.tar.gz /usr/local/java
cd /usr/local/java
tar -zxvf jdk-12.0.2_linux-x64_bin.tar.gz 
rm jdk-12.0.2_linux-x64_bin.tar.gz
```
配置环境变量，并生效
```Bash
vi /etc/profile
source /etc/profile
```
添加内容为
```Bash
#java environment
export JAVA_HOME=/usr/local/java/jdk-12.0.2
export CLASSPATH=.:$JAVA_HOME/jre/lib/rt.jar:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
export PATH=$PATH:$JAVA_HOME/bin
```
查看是否配置成功
```Bash
java -version
java version "12.0.2" 2019-07-16
Java(TM) SE Runtime Environment (build 12.0.2+10)
Java HotSpot(TM) 64-Bit Server VM (build 12.0.2+10, mixed mode, sharing)
```

## 5. 熟悉linux 常用命令
```Bash
ls # 列出目录下文件
mv # 移动文件或文件夹
mkdir # 创建文件夹
vi # 编辑文本
cd # 移动到某目录
wget # 下载某链接里的内容
......
```

## 6. 熟悉shell 变量/循环/条件判断/函数等
```Bash
vi fun_6th.sh
```
在其中写入
```Bash
#! /bin/bash

for ((i=1; i<=100; i ++));
do
 echo $i >> output.txt
 rem=$(( $i % 10 ))
 if [[ $rem == 0 ]];then
 echo $i >> output.txt
 fi
done
```






