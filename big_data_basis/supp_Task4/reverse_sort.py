#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: reverse_sort.py
# @time: 2019/8/3 下午10:41

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

'''
python reverse_sort.py -r local -o reverse_sort ./movielens/ml-100k/u.data
'''
