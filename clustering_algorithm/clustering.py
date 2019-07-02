#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: clustering.py
# @time: 2019/7/2 10:02

from time import time
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score


def c_means(data, num_cluster, label=None, tol=1e-10, max_iter=100):
    # C means algorithm
    # sample initial cluster centers randomly
    centers = data.sample(n=num_cluster, random_state=123, axis=0)
    print('initial cluster centers:')
    print(centers)
    # iteration
    new_centers = centers
    for idx_iter in range(max_iter):
        centers = new_centers.copy()
        res_cluster = []
        for i in data.index:
            distance = [
                np.linalg.norm(data.iloc[i, :] - centers.iloc[j, :])
                for j in range(num_cluster)]
            res_cluster.append(np.argmin(distance))
        res_cluster = pd.Series(res_cluster)
        # update centers
        new_centers = centers.copy()
        for j in range(num_cluster):
            new_centers.iloc[j, :] = np.mean(data.loc[res_cluster == j, :])
        # termination condition
        diff = np.sum(
            [np.linalg.norm(new_centers.iloc[i, :] - centers.iloc[i, :])
             for i in range(num_cluster)])
        if diff < tol:
            break
        if idx_iter == max_iter - 1:
            warnings.warn('Cannot converge until maximum iterations')

    print("Iterations: ", idx_iter)

    # error rate
    if label is not None:
        dict_cluster_type = {}
        for j in range(num_cluster):
            sub_data = data.loc[label == j, 'cluster']
            mode = stats.mode(sub_data)[0][0]
            dict_cluster_type[mode] = j
        res_cluster = [dict_cluster_type[i] for i in res_cluster]
        error_rate = 1 - accuracy_score(label, res_cluster)
        print("Error rate of clustering: ", error_rate)

    # output clustering results
    if label is not None:
        result = pd.concat([data, label, res_cluster], axis=1)
    else:
        result = pd.concat([data, res_cluster], axis=1)

    result.to_csv('.//clustering_algorithm//results_cmeans.txt', sep='\t')

    return


def mmd(data, num_cluster=None, theta=None, label=None, select_centers=False):
    # MMD algorithm
    # sample a initial cluster center randomly
    center = data.sample(n=1, random_state=123, axis=0)

    # select second center
    distance2 = [np.linalg.norm(data.iloc[i, :] - center.iloc[0, :])
                 for i in data.index]
    center = center.append(data.iloc[np.argmax(distance2), :])
    d12 = np.max(distance2)

    # select more centers
    if num_cluster is not None and theta is not None:
        print('Error: too many inputs')
    elif num_cluster is not None:
        for j in range(2, num_cluster):
            distance = \
                [min(np.linalg.norm(data.iloc[i, :] - center.iloc[:, :],
                                    axis=1))
                 for i in data.index]
            center = center.append(data.iloc[np.argmax(distance), :])
    elif theta is not None:
        max_dis = d12
        while max_dis > d12*theta:
            distance = \
                [min(np.linalg.norm(data.iloc[i, :] - center.iloc[:, :],
                                    axis=1))
                 for i in data.index]
            center = center.append(data.iloc[np.argmax(distance), :])
    else:
        print("Error: too few inputs")

    if select_centers:
        return center
    else:
        # clustering
        res_cluster = []
        for i in data.index:
            distance = [
                np.linalg.norm(data.iloc[i, :] - center.iloc[j, :])
                for j in range(center.shape[1])]
            res_cluster.append(np.argmin(distance))
        res_cluster = pd.Series(res_cluster)

        # error rate
        if label is not None:
            dict_cluster_type = {}
            for j in range(num_cluster):
                sub_data = data.loc[label == j, 'cluster']
                mode = stats.mode(sub_data)[0][0]
                dict_cluster_type[mode] = j
            res_cluster = [dict_cluster_type[i] for i in res_cluster]
            error_rate = 1 - accuracy_score(label, res_cluster)
            print("Error rate of clustering: ", error_rate)

        # output clustering results
        if label is not None:
            result = pd.concat([data, label, res_cluster], axis=1)
        else:
            result = pd.concat([data, res_cluster], axis=1)

        result.to_csv('.//clustering_algorithm//results_cmeans.txt', sep='\t')

        return


if __name__ == '__main__':
    time_start = time()
    # import data
    iris = pd.read_csv(".//clustering_algorithm//iris.dat", sep='\t',
                       header=None)
    iris.columns = ['V1', 'V2', 'V3', 'V4', 'label']
    iris_label = iris['label']
    iris_data = iris.iloc[:, 0:4]
    # c_means clustering
    c_means(iris_data, 3)

    #

    time_end = time()
    print(time_end - time_start)
