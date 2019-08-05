#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: sort.py
# @time: 2019/8/3 下午10:23

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

'''
python sort.py -r local -o sort ./movielens/ml-100k/u.data
'''
