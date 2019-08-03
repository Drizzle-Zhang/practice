# 【Task4】MapReduce+MapReduce执行过程

## 1. MR原理
见[Task2](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/Task2.md)第3部分

## 2. 使用Hadoop Streaming -python写出WordCount
见[Task2](https://github.com/Drizzle-Zhang/practice/blob/master/big_data_basis/Task2.md)第6部分

## 3. 使用mr计算movielen中每个用户的平均评分。
先下载movielen数据集（[下载链接](http://files.grouplens.org/datasets/movielens/)）
```Bash
mkdir movielens
cd movielens
wget http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```
安装mrjob包
```Bash
conda install -c conda-forge mrjob
```
写出使用mr计算movielen中每个用户的平均评分的python脚本
```Bash
#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: user_ave.py
# @time: 2019/8/3 下午5:46
from abc import ABC

from mrjob.job import MRJob
import numpy as np


class UserAverageRating(MRJob, ABC):
    def mapper(self, key, line):
        # 接收每一行的输入数据，处理后返回一堆key:value，即user：rating
        list_line = line.strip().split('::')
        user = list_line[0]
        rating = int(list_line[2])
        yield user, rating

    def reducer(self, user, ratings):
        # 接收mapper输出的key:value对进行整合，把相同key的value做累加（sum）操作后输出
        yield user, np.mean(list(ratings))


if __name__ == '__main__':
    UserAverageRating.run()
```
在本地运行该脚本，并查看结果
```Bash
cd ~/practice/big_data_basis/supp_Task4
python user_ave.py -r local -o user_ave ./movielens/ml-1m/ratings.dat

ls user_ave
part-00000  part-00002  part-00004  part-00006
part-00001  part-00003  part-00005  part-00007

head -6 user_ave/part-00000
"1"	4.1886792453
"10"	4.114713217
"100"	3.0263157895
"1000"	4.130952381
"1001"	3.6525198939
"1002"	4.1363636364

```





