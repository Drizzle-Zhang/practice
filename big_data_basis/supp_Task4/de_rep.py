#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: de_rep.py
# @time: 2019/8/3 下午9:59

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

'''
python de_rep.py -r local -o de_rep ./movielens/ml-100k/u.data
'''
