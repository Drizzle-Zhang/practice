#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: pagerank.py
# @time: 2019/8/4 上午12:59

from mrjob.job import MRJob
import numpy as np
from sklearn.metrics import jaccard_score


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

'''
python pagerank.py -r local -o pagerank ./input_page.txt
'''
