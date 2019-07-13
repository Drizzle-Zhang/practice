#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: LLE.py
# @time: 2019/7/11 21:27

from time import time
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Eigenface.eigenface import EigenFaceAndFisher
from sklearn.neighbors import NearestNeighbors


def lle(df_train, info_train, k_knn=4, k_pca=100, k_lle=2):
    # locally linear embedding algorithm
    obj_pca = EigenFaceAndFisher(df_train, info_train)
    vec_project, red_pca = \
        obj_pca.get_eigen_projection(k_pca=k_pca)
    neigh = NearestNeighbors(n_neighbors=k_knn)
    neigh.fit = neigh.fit(red_pca)
    dist_neigh, neighbors = neigh.kneighbors()
    n_sample = df_train.shape[0]
    list_w = []
    for i_lle in range(n_sample):
        c_i = np.dot(
            red_pca[i_lle, :] - red_pca[neighbors[i_lle, :], :],
            (red_pca[i_lle, :] - red_pca[neighbors[i_lle, :], :]).T
        )
        w_i = np.sum(c_i, axis=1) / np.sum(np.sum(c_i))
        list_w.append(w_i)
    array_w = np.array(list_w)
    w_sparse = np.zeros((n_sample, n_sample))
    for i_w in range(n_sample):
        for k_w in range(k_knn):
            j_w = neighbors[i_w, k_w]
            w_sparse[i_w, j_w] = array_w[i_w, k_w]
    mat = np.dot(
        (np.eye(n_sample) - w_sparse).T, np.eye(n_sample) - w_sparse
    )
    eigval_lle, u_lle = np.linalg.eig(mat)
    sort_eigval_lle = np.argsort(eigval_lle)
    mat_lle = u_lle[:, sort_eigval_lle[:k_lle]]

    return mat_lle


if __name__ == '__main__':
    time_start = time()
    # load images
    dir_images = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//ORL'
    images = os.listdir(dir_images)
    images = [image for image in images if image[:3] == 'orl']
    images_info = []
    df_images = pd.DataFrame()
    for i in range(len(images)):
        image = images[i]
        obj_image = Image.open(os.path.join(dir_images, image))
        vec_image = np.matrix(obj_image.getdata()).tolist()
        df_images = df_images.append(vec_image)
        if i % 10 < 5:
            images_info.append({'person_id': i // 10 + 1, 'filename': image,
                                'group': 'train', 'photo_id': i % 10 + 1})
        else:
            images_info.append({'person_id': i // 10 + 1, 'filename': image,
                                'group': 'test', 'photo_id': i % 10 + 1})
    df_info = pd.DataFrame(images_info)
    df_info.index = images
    df_images.index = images

    # split data set
    info_trainset = df_info.loc[df_info['group'] == 'train', :]
    info_testset = df_info.loc[df_info['group'] == 'test', :]
    df_trainset = df_images.loc[info_trainset['filename'], :]
    df_testset = df_images.loc[info_testset['filename'], :]

    # reduction results
    mat_train = lle(df_trainset, info_trainset)
    mat_test = lle(df_testset, info_testset)

    # plot train set
    lle_train = np.array(mat_train)
    path = 'C://Users//zhangyu//Documents//my_git//practice//LLE//'
    fig_lle_train, ax_lle_train = plt.subplots()
    ax_lle_train.plot(lle_train[:, 0], lle_train[:, 1], 'ok')
    ax_lle_train.set_xlabel('Dimension 1')
    ax_lle_train.set_ylabel('Dimension 2')
    fig_lle_train.savefig(
        os.path.join(path, 'LLE_train.png')
    )

    # compare train set with test set
    df_mat_train = pd.DataFrame(mat_train, index=df_trainset.index)
    df_mat_test = pd.DataFrame(mat_test, index=df_testset.index)
    persons = set(np.array(info_trainset['person_id']).tolist())
    df_ave = pd.DataFrame()
    for person in persons:
        vec_person = np.mean(
            df_mat_train.loc[info_trainset['person_id'] == person, :])
        vec_person.name = person
        df_ave = df_ave.append(vec_person)
    array_test = np.array(df_mat_test)
    array_ave = np.array(df_ave)
    distances = np.linalg.norm(
        array_test[:, np.newaxis, :] - array_ave[np.newaxis, :, :], axis=-1)
    labels = np.argmin(distances, axis=1) + 1
    accuracy = accuracy_score(info_testset['person_id'], labels)

    print("Accuracy of minimum distance algorithm: ", accuracy)

    # result file
    output = info_testset.loc[:, ['filename', 'person_id']]
    output.columns = ['Sample ID', 'Known cluster']
    output['Identified cluster'] = labels
    file_out = os.path.join(path, 'classification.txt')
    output.to_csv(file_out, sep='\t', index=None)

    time_end = time()
    print(time_end - time_start)
