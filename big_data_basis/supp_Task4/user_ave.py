#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: user_ave.py
# @time: 2019/8/3 下午5:46

from mrjob.job import MRJob
import numpy as np


class UserAverageRating(MRJob):
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

'''
python user_ave.py -r local -o user_ave ./movielens/ml-1m/ratings.dat
'''
