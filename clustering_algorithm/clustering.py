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
import matplotlib.pyplot as plt


def c_means(data, path_result=None, num_cluster=None, label=None, centers=None,
            tol=1e-10, max_iter=100, return_error_rate=False):
    # C means algorithm
    if centers is not None and num_cluster is not None:
        print('Error: too many inputs')
    elif centers is not None:
        # given centers
        centers = centers
    elif num_cluster is not None:
        # sample initial cluster centers randomly
        centers = data.sample(n=num_cluster, random_state=123, axis=0)
    else:
        print("Error: too few inputs")

    print('initial cluster centers:')
    print(centers)
    num_cluster = centers.shape[0]
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
            sub_res = label.loc[res_cluster == j]
            mode = stats.mode(sub_res)[0][0]
            dict_cluster_type[mode] = j
        res_cluster = pd.Series([dict_cluster_type[i] for i in res_cluster],
                                name='cluster')
        error_rate = 1 - accuracy_score(label, res_cluster)
        print("Error rate of clustering: ", error_rate)

    # output clustering results
    if label is not None:
        result = pd.concat([data, label, res_cluster], axis=1)
    else:
        result = pd.concat([data, res_cluster], axis=1)

    if path_result is not None:
        result.to_csv(path_result, sep='\t')

    if return_error_rate and label is not None:
        return error_rate

    return


def mmd(data, num_cluster=None, theta=None, label=None, select_centers=False,
        path_result=None):
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
            max_dis = max(distance)
            center = center.append(data.iloc[np.argmax(distance), :])
    else:
        print("Error: too few inputs")

    if select_centers:
        return center
    else:
        print('cluster centers:')
        print(center)
        num_cluster = center.shape[0]
        # clustering
        res_cluster = []
        for i in data.index:
            distance = [
                np.linalg.norm(data.iloc[i, :] - center.iloc[j, :])
                for j in range(center.shape[0])]
            res_cluster.append(np.argmin(distance))
        res_cluster = pd.Series(res_cluster, name='cluster')

        # error rate
        if label is not None:
            dict_cluster_type = {}
            for j in range(num_cluster):
                sub_res = res_cluster.loc[label == j]
                mode = stats.mode(sub_res)[0][0]
                dict_cluster_type[mode] = j
            res_cluster = pd.Series(
                [dict_cluster_type[i] for i in res_cluster],
                name='cluster')
            error_rate = 1 - accuracy_score(label, res_cluster)
            print("Error rate of clustering: ", error_rate)

        # output clustering results
        if label is not None:
            result = pd.concat([data, label, res_cluster], axis=1)
        else:
            result = pd.concat([data, res_cluster], axis=1)

        result.to_csv(path_result, sep='\t')

        return


def spectral_clustering(data, k):
    # get a spectral matrix, then execute c_means
    # similarity matrix
    data_array = np.array(data)
    sim_mat = np.exp(
        -0.5*np.linalg.norm(
            data_array[:, np.newaxis, :] - data_array[np.newaxis, :, :],
            axis=-1))

    vec_diag = np.sum(sim_mat, axis=1)
    diag = np.eye(len(vec_diag))
    for i in range(len(vec_diag)):
        diag[i, i] = vec_diag[i]
    l_mat = diag - sim_mat

    lamda, u_mats = np.linalg.eig(l_mat)
    sort_lambda = np.argsort(lamda)
    u_mat = u_mats[:, sort_lambda[:k]]

    return u_mat


if __name__ == '__main__':
    time_start = time()
    # import data
    iris = pd.read_csv(
        'C://Users//zhangyu//Documents//my_git//practice//'
        'clustering_algorithm//iris.dat', sep='\t', header=None)
    iris.columns = ['V1', 'V2', 'V3', 'V4', 'label']
    iris_label = iris['label']
    iris_data = iris.iloc[:, 0:4]
    # c_means clustering
    """
    c_means(
        iris_data, num_cluster=3,
        path_result='C://Users//zhangyu//Documents//my_git//practice//'
        'clustering_algorithm//results_cmeans.txt', label=iris_label)"""

    # select centers by MMD then c_means
    """
    iris_centers = mmd(iris_data, num_cluster=3, select_centers=True)
    c_means(
        iris_data, centers=iris_centers,
        path_result='C://Users//zhangyu//Documents//my_git//practice//'
                    'clustering_algorithm//results_cmeans.txt',
        label=iris_label)"""

    # MMD clustering
    """
    mmd(iris_data, theta=0.8, label=iris_label,
        path_result='C://Users//zhangyu//Documents//my_git//practice//'
                    'clustering_algorithm//results_mmd.txt')"""

    # spectral clustering
    list_error_rate = []
    for idx in range(1, 40):
        iris_u_mat = spectral_clustering(iris_data, idx)
        iris_u_mat = pd.DataFrame(iris_u_mat)
        iris_error_rate = c_means(
            iris_u_mat, num_cluster=3, label=iris_label,
            return_error_rate=True)
        list_error_rate.append(iris_error_rate)
    plt.plot(range(1, 40), list_error_rate, '-ok')
    plt.xlabel('Numbers of eigenvectors')
    plt.ylabel('Error rate')
    plt.show()
    opt_idx = np.argmin(list_error_rate)
    iris_u_mat = spectral_clustering(iris_data, opt_idx + 1)
    iris_u_mat = pd.DataFrame(iris_u_mat)
    c_means(iris_u_mat, num_cluster=3, label=iris_label,
            path_result='C://Users//zhangyu//Documents//my_git//practice//'
                        'clustering_algorithm//results_spec_clustering.txt')

    time_end = time()
    print('Run time: ', time_end - time_start)
