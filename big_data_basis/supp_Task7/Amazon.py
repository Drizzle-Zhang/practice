#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: Amazon.py
# @time: 8/10/19 6:00 PM

from time import time
import json
from pyspark import SparkConf, SparkContext


def dict2list(line):
    dict_line = json.loads(line)
    list_line = [dict_line["reviewerID"],
                 dict_line["overall"], dict_line["reviewText"]]

    return list_line


def calc_tf(list_line):
    list_text = list_line[2].split(" ")
    cnt_map = {}
    for w in list_text:
        cnt_map[w] = cnt_map.get(w, 0) + 1
    total_w = len(list_text)

    return [(list_line[0], (w, float(cnt)/total_w))
            for w, cnt in cnt_map.items()]


def


if __name__ == '__main__':
    time_start = time()

    time_end = time()
    print(time_end - time_start)