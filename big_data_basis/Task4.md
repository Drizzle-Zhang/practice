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

## 4. 使用mr实现merge功能。根据item，merge movielen中的 u.data u.item
写出使用mr实现merge功能的python脚本
```Python
from mrjob.job import MRJob


class Merge(MRJob):
    def mapper(self, _, line):
        list_tab = line.strip().split('\t')
        list_tube = line.strip().split('|')
        if len(list_tab) > 1 and len(list_tube) == 1:
            item_id = list_tab[1]
            user_info = line
            yield item_id, user_info
        elif len(list_tube) > 1 and len(list_tab) == 1:
            item_id = list_tube[0]
            item_info = line
            yield item_id, item_info

    def reducer(self, item_id, info):
        yield item_id, info


if __name__ == '__main__':
    Merge.run()

```
在终端运行该脚本，并查看运行结果
```Bash
python merge.py -r local -o merge ./movielens/ml-100k/u.data ./movielens/ml-100k/u.item 
ls merge
part-00000  part-00002  part-00004  part-00006
part-00001  part-00003  part-00005  part-00007
head -1 merge/part-00000
（结果太长）
```

## 5. 使用mr实现去重任务。
去重任务的具体形式是，在u.data文件中查看每个user都做出了哪些评级
Python脚本如下
```Python
from mrjob.job import MRJob


class DeRepetition(MRJob):
    def mapper(self, _, line):
        list_tab = line.strip().split('\t')
        user_id = list_tab[0]
        rating = list_tab[2]
        yield user_id, rating

    def reducer(self, user_id, rating):
        yield user_id, set(rating)


if __name__ == '__main__':
    DeRepetition.run()

```
运行脚本并查看结果
```Bash
python de_rep.py -r local -o de_rep ./movielens/ml-100k/u.data

ls de_rep
part-00000  part-00002  part-00004  part-00006
part-00001  part-00003  part-00005  part-00007

head de_rep/part-00000
"1"	["2","3","1","4","5"]
"10"	["5","4","3"]
"100"	["2","3","1","4","5"]
"101"	["2","3","1","4","5"]
"102"	["1","2","4","3"]
"103"	["2","3","1","4","5"]
"104"	["2","3","1","4","5"]
"105"	["5","2","4","3"]
"106"	["5","2","4","3"]
"107"	["2","3","1","4","5"]

```

## 6. 使用mr实现排序。
排序任务的具体情形是，在u.data文件中查看每个user都做出了哪些评级，并对去重过的评级进行排序
```Python
from mrjob.job import MRJob
import numpy as np


class Sort(MRJob):
    def mapper(self, _, line):
        list_tab = line.strip().split('\t')
        user_id = list_tab[0]
        rating = int(list_tab[2])
        yield user_id, rating

    def reducer(self, user_id, rating):
        ratings = list(set(rating))
        yield user_id, np.sort(ratings).tolist()


if __name__ == '__main__':
    Sort.run()

```

```Bash
python sort.py -r local -o sort ./movielens/ml-100k/u.data

ls sort
part-00000  part-00002  part-00004  part-00006
part-00001  part-00003  part-00005  part-00007

head sort/part-00000
"1"	[1,2,3,4,5]
"10"	[3,4,5]
"100"	[1,2,3,4,5]
"101"	[1,2,3,4,5]
"102"	[1,2,3,4]
"103"	[1,2,3,4,5]
"104"	[1,2,3,4,5]
"105"	[2,3,4,5]
"106"	[2,3,4,5]
"107"	[1,2,3,4,5]

```

## 7. 使用mapreduce实现倒排索引。
倒排索引任务的具体情形是，在u.data文件中查看每个user都做出了哪些评级，并得到去重过的评级的倒排索引
```Python
from mrjob.job import MRJob
import numpy as np


class ReverseSort(MRJob):
    def mapper(self, _, line):
        list_tab = line.strip().split('\t')
        user_id = list_tab[0]
        rating = int(list_tab[2])
        yield user_id, rating

    def reducer(self, user_id, rating):
        ratings = list(set(rating))
        index_rev = np.argsort(ratings)[::-1]
        yield user_id, index_rev.tolist()


if __name__ == '__main__':
    ReverseSort.run()

```

```Bash
python reverse_sort.py -r local -o reverse_sort ./movielens/ml-100k/u.data

head reverse_sort/part-00000
"1"	[4,3,2,1,0]
"10"	[2,1,0]
"100"	[4,3,2,1,0]
"101"	[4,3,2,1,0]
"102"	[3,2,1,0]
"103"	[4,3,2,1,0]
"104"	[4,3,2,1,0]
"105"	[3,2,1,0]
"106"	[3,2,1,0]
"107"	[4,3,2,1,0]

```

## 8. 使用mapreduce计算Jaccard相似度。
计算的具体情形是，对u.item里的影片类型进行Jaccard相似度计算,判断各个电影和电影1的相似度
```Python
from mrjob.job import MRJob
import numpy as np
from sklearn.metrics import jaccard_score


class Jaccard(MRJob):
    def mapper(self, _, line):
        list_tube = line.strip().split('|')
        item_id = list_tube[0]
        item_type = list_tube[5:]
        yield item_id, item_type

    def reducer(self, item_id, item_type):
        ref_type = \
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        item_type0 = item_type
        for item_type in item_type0:
            item_type = item_type
        item_type = [int(num) for num in item_type]
        if len(item_type) == len(ref_type):
            score = jaccard_score(ref_type, np.array(item_type))
            yield item_id, score


if __name__ == '__main__':
    Jaccard.run()

```

```Bash
python Jaccard.py -r local -o Jaccard ./movielens/ml-100k/u.item

head Jaccard/part-00000
"1"	1.0
"10"	0.0
"100"	0.0
"1000"	0.25
"1001"	0.3333333333
"1002"	0.3333333333
"1003"	0.5
"1004"	0.0
"1005"	0.0
"1006"	0.0

```

## 9. 使用mapreduce实现PageRank。
```Python
class PageRank(MRJob):
    def mapper(self, _, line):
        list_line = line.strip().split(' ')
        node0 = list_line[0]
        yield node0, 1

    def reducer(self, node, recurrence):
        n = 3
        n_p = 4
        alpha = 0.8
        values = alpha * sum(recurrence)/n + (1 - alpha)/n_p

        yield node, values


if __name__ == '__main__':
    PageRank.run()

```
```Bash
python pagerank.py -r local -o pagerank ./input_page.txt

head pagerank/part-00000
"A"	0.85

head pagerank/part-00001
"B"	0.5833333333

```






