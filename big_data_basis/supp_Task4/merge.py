#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: merge.py
# @time: 2019/8/3 下午8:55

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

'''
python merge.py -r local -o merge ./movielens/ml-100k/u.data 
./movielens/ml-100k/u.item 
'''
