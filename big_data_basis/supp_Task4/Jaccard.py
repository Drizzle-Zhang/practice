#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: Jaccard.py
# @time: 2019/8/3 下午10:53

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

'''
python Jaccard.py -r local -o Jaccard ./movielens/ml-100k/u.item
'''
