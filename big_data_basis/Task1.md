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



