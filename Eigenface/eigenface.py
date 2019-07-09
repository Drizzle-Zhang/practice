#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: eigenface.py
# @time: 2019/7/3 15:39

from time import time
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import accuracy_score


def min_distance(df_mat, info_mat, file_out):
    # minimum distance algorithm
    # average vector
    info_train = info_mat.loc[info_mat['group'] == 'train', :]
    info_test = info_mat.loc[info_mat['group'] == 'test', :]
    df_train = df_mat.loc[info_train['filename'], :]
    df_test = df_mat.loc[info_test['filename'], :]
    persons = set(np.array(info_mat['person_id']).tolist())
    df_ave = pd.DataFrame()
    for person in persons:
        vec_person = np.mean(
            df_train.loc[info_train['person_id'] == person, :])
        vec_person.name = person
        df_ave = df_ave.append(vec_person)
    array_test = np.array(df_test)
    array_ave = np.array(df_ave)
    distances = np.linalg.norm(
        array_test[:, np.newaxis, :] - array_ave[np.newaxis, :, :], axis=-1)
    labels = np.argmin(distances, axis=1) + 1
    accuracy = accuracy_score(info_test['person_id'], labels)

    print("Accuracy of minimum distance algorithm: ", accuracy)

    # result file
    output = info_test.loc[:, ['filename', 'person_id']]
    output.columns = ['Sample ID', 'Known cluster']
    output['Identified cluster'] = labels
    output.to_csv(file_out, sep='\t', index=None)

    return


def pca_reduction(df_mat, info_mat, path_result, k, output_mat=True):
    # split data set
    info_train = info_mat.loc[info_mat['group'] == 'train', :]
    info_test = info_mat.loc[info_mat['group'] == 'test', :]
    df_train = df_mat.loc[info_train['filename'], :]
    df_test = df_mat.loc[info_test['filename'], :]

    # normalization
    ave_train = np.mean(df_train)
    phi_train = df_train - ave_train
    ave_test = np.mean(df_test)
    phi_test = df_test - ave_test

    # cov
    cov_train = np.dot(phi_train.T, phi_train) / phi_train.shape[0]
    # cov_train2 = np.cov(df_train.T)

    # calculate eigenvalue
    vec_lambda, u_mat = np.linalg.eig(cov_train)
    sort_lambda = np.argsort(vec_lambda)[::-1]

    if output_mat:
        for one_k in k:
            # dimension reduction
            vec_project = u_mat[:, sort_lambda[:one_k]]
            vec_project = np.real(vec_project)
            red_train = np.dot(df_train, vec_project)
            red_train = pd.DataFrame(red_train, index=df_train.index)
            red_train.to_csv(
                os.path.join(path_result, f'Reduction_train_{one_k}.txt'),
                sep='\t'
            )

            # error of train
            red_phi_train = np.dot(red_train, vec_project.T)
            red_phi_train = pd.DataFrame(red_phi_train, index=df_train.index)
            red_train.to_csv(
                os.path.join(path_result, f'Reconstruction_train_{one_k}.txt'),
                sep='\t'
            )
            error_train = np.sum(np.sum(phi_train - red_phi_train))
            print("Error of train set: ", error_train)

            # dimension reduction
            red_test = np.dot(df_test, vec_project)
            red_test = pd.DataFrame(red_test, index=df_test.index)
            red_test.to_csv(
                os.path.join(path_result, f'Reduction_test_{one_k}.txt'),
                sep='\t'
            )

            # error of test
            red_phi_test = np.dot(red_test, vec_project.T)
            red_phi_test = pd.DataFrame(red_phi_test, index=df_test.index)
            red_phi_test.to_csv(
                os.path.join(path_result, f'Reconstruction_test_{one_k}.txt'),
                sep='\t'
            )
            error_test = np.sum(np.sum(phi_test - red_phi_test))
            print("Error of test set: ", error_test)

    else:
        list_error_rate = []
        for one_k in k:
            # dimension reduction
            vec_project = u_mat[:, sort_lambda[:one_k]]
            vec_project = np.real(vec_project)
            red_train = np.dot(df_train, vec_project)

            # error of train
            red_phi_train = np.dot(red_train, vec_project.T)
            error_train = np.sum(np.sum(phi_train - red_phi_train))

            # dimension reduction
            red_test = np.dot(df_test, vec_project)

            # error of test
            red_phi_test = np.dot(red_test, vec_project.T)
            error_test = np.sum(np.sum(phi_test - red_phi_test))

            list_error_rate.append(
                {'k': one_k, 'error_train': error_train,
                 'error_test': error_test}
            )
        df_error_rate = pd.DataFrame(list_error_rate)
        df_error_rate.to_csv(
            os.path.join(path_result, f"Error_rate_{k[0]}-{k[-1]}.txt"),
            sep='\t'
        )

    return


def display_image(vector_image, path_file):
    # reshape
    mat_image = vector_image.reshape((112, 92))
    mat_image = np.uint8(mat_image)
    im = Image.fromarray(mat_image)
    im.save(path_file)

    return


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

    # minimum distance algorithm
    min_distance(
        df_images, df_info,
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'result_min_distance.txt')

    # PCA reduction and construction
    path = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
    pca_reduction(df_images, df_info, path, k=[50, 100, 200, 300])

    # average face of train set
    info_train = df_info.loc[df_info['group'] == 'train', :]
    df_train = df_images.loc[info_train['filename'], :]
    ave_train = np.mean(df_train)
    display_image(
        ave_train,
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Average_face_train.png'
    )

    pca_reduction(df_images, df_info, path, k=range(50, 201), output_mat=False)

    time_end = time()
    print(time_end - time_start)
