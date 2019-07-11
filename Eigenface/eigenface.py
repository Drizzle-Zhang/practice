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
import matplotlib.pyplot as plt


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
            print("k = ", one_k)
            # dimension reduction
            vec_project = u_mat[:, sort_lambda[:one_k]]
            vec_project = np.real(vec_project)
            red_train = np.dot(phi_train, vec_project)
            """
            red_train = pd.DataFrame(red_train, index=df_train.index)
            red_train.to_csv(
                os.path.join(path_result, f'Reduction_train_{one_k}.txt'),
                sep='\t'
            )"""

            # error of train
            red_phi_train = np.dot(red_train, vec_project.T)
            red_df_train = red_phi_train + np.array(ave_train)
            red_df_train = pd.DataFrame(red_df_train, index=df_train.index)
            red_df_train.to_csv(
                os.path.join(path_result, f'Reconstruction_train_{one_k}.txt'),
                sep='\t'
            )
            error_train = np.sum(np.linalg.norm(
                np.array(df_train) - np.array(red_df_train), axis=-1))
            print("Error of train set: ", error_train)

            # dimension reduction
            red_test = np.dot(phi_test, vec_project)
            """
            red_test = pd.DataFrame(red_test, index=df_test.index)
            red_test.to_csv(
                os.path.join(path_result, f'Reduction_test_{one_k}.txt'),
                sep='\t'
            )
            """

            # error of test
            red_phi_test = np.dot(red_test, vec_project.T)
            red_df_test = red_phi_test + np.array(ave_test)
            red_df_test = pd.DataFrame(red_df_test, index=df_test.index)
            red_df_test.to_csv(
                os.path.join(path_result, f'Reconstruction_test_{one_k}.txt'),
                sep='\t'
            )
            error_test = np.sum(np.linalg.norm(
                np.array(df_test) - np.array(red_df_test), axis=-1))
            print("Error of test set: ", error_test)

    else:
        list_error_rate = []
        for one_k in k:
            # dimension reduction
            vec_project = u_mat[:, sort_lambda[:one_k]]
            vec_project = np.real(vec_project)
            red_train = np.dot(phi_train, vec_project)

            # error of train
            red_phi_train = np.dot(red_train, vec_project.T)
            red_df_train = red_phi_train + np.array(ave_train)
            error_train = np.sum(np.linalg.norm(
                np.array(df_train) - np.array(red_df_train), axis=-1))

            # dimension reduction
            red_test = np.dot(phi_test, vec_project)

            # error of test
            red_phi_test = np.dot(red_test, vec_project.T)
            red_df_test = red_phi_test + np.array(ave_test)
            error_test = np.sum(np.linalg.norm(
                np.array(df_test) - np.array(red_df_test), axis=-1))

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
    vector_image = np.array(vector_image)
    mat_image = vector_image.reshape((112, 92))
    mat_image = np.uint8(mat_image)
    im = Image.fromarray(mat_image)
    im.save(path_file)

    return


class EigenFaceAndFisher:
    # eigenface algorithm
    def __init__(self, df_train, info_train):
        self.df_train = df_train
        self.info_train = info_train
        # centralization
        self.ave_train = np.mean(df_train)
        self.phi_train = df_train - self.ave_train

        # cov
        self.cov_train = np.dot(
            self.phi_train.T, self.phi_train) / self.phi_train.shape[0]

        # calculate eigenvalue
        self.vec_lambda, self.u_mat = np.linalg.eig(self.cov_train)
        self.sort_lambda = np.argsort(self.vec_lambda)[::-1]

        # create the register
        self.persons = set(np.array(info_train['person_id']).tolist())
        self.df_ave = pd.DataFrame()
        self.num_photo_person = []
        for person in self.persons:
            vec_person = np.mean(
                self.phi_train.loc[info_train['person_id'] == person, :]
            )
            vec_person.name = person
            self.num_photo_person.append(
                self.phi_train.loc[
                info_train['person_id'] == person, :].shape[0]
            )
            self.df_ave = self.df_ave.append(vec_person)

    def get_eigen_projection(self, k_pca):
        # eigenfacer projection matrix
        vec_project = self.u_mat[:, self.sort_lambda[:k_pca]]
        pca_vec_project = np.real(vec_project)

        # PCA reduction
        red_pca_train = np.dot(self.phi_train, vec_project)

        return pca_vec_project, red_pca_train

    def get_fisher_projection(self, k_pca, k_lda):
        # fisher projection matrix
        pca_vec_project, red_pca_train = self.get_eigen_projection(k_pca)
        # centers
        mu_total = np.mean(red_pca_train)
        red_pca_train = pd.DataFrame(red_pca_train, index=self.df_train.index)
        mu_persons = pd.DataFrame()
        for person in self.persons:
            vec_person = np.mean(
                red_pca_train.loc[self.info_train['person_id'] == person, :])
            vec_person.name = person
            mu_persons = mu_persons.append(vec_person)

        # calculate scatter matrix
        s_w = np.zeros((pca_vec_project.shape[1], pca_vec_project.shape[1]))
        s_b = np.zeros((pca_vec_project.shape[1], pca_vec_project.shape[1]))
        for idx in range(len(self.persons)):
            s_w = s_w + np.dot(
                (red_pca_train - mu_persons.iloc[idx, :]).T,
                red_pca_train - mu_persons.iloc[idx, :]
            )
            s_b = s_b + self.num_photo_person[idx] * np.dot(
                (mu_persons.iloc[idx, :] - mu_total).T,
                mu_persons.iloc[idx, :] - mu_total
            )

        # eigenvalue and eigenvector
        s_w_inv = np.linalg.inv(s_w)
        s_w_b = np.dot(s_w_inv, s_b)
        eigval_lda, eig_vec_lda = np.linalg.eig(s_w_b)
        sort_eigval_lda = np.argsort(eigval_lda)[::-1]

        # LDA reduction
        lda_project = eig_vec_lda[:, sort_eigval_lda[:k_lda]]
        lda_vec_project = np.real(lda_project)

        # PCA reduction
        red_lda_train = np.dot(red_pca_train, lda_vec_project)

        return lda_vec_project, red_lda_train

    def predict_label(self, df_test, pca_vec_project, lda_vec_project):
        # centralization
        ave_test = np.mean(df_test)
        phi_test = df_test - ave_test
        red_test = np.dot(np.dot(phi_test, pca_vec_project), lda_vec_project)

        # predict labels
        df_regi = np.dot(np.dot(self.df_ave, pca_vec_project), lda_vec_project)

        # minimum distance
        array_test = np.array(red_test)
        array_regi = np.array(df_regi)
        distances = np.linalg.norm(
            array_test[:, np.newaxis, :] - array_regi[np.newaxis, :, :],
            axis=-1)
        labels = np.argmin(distances, axis=1) + 1

        return labels


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
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//'
    pca_reduction(df_images, df_info, path, k=[50, 100, 200, 300])

    # average face of train set
    info_train_ave = df_info.loc[df_info['group'] == 'train', :]
    df_train_ave = df_images.loc[info_train_ave['filename'], :]
    ave_face = np.mean(df_train_ave)
    display_image(
        ave_face,
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Average_face_train.png'
    )

    # k=50
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_train_50.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_train_50.png'
    )
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_test_50.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_test_50.png'
    )

    # k=100
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_train_100.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_train_100.png'
    )
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_test_100.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_test_100.png'
    )

    # k=200
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_train_200.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_train_200.png'
    )
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_test_200.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_test_200.png'
    )

    # k=300
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_train_300.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_train_300.png'
    )
    file_rec = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//' \
        'results_reduction_construction//Reconstruction_test_300.txt'
    df_rec = pd.read_csv(file_rec, sep='\t', index_col=0)
    display_image(
        df_rec.iloc[0, :],
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
        'Reconstruction_face_test_300.png'
    )

    # Curve of error with the change of k
    path = \
        'C://Users//zhangyu//Documents//my_git//practice//Eigenface//'
    pca_reduction(df_images, df_info, path, k=range(1, 500), output_mat=False)
    df_k_error = pd.read_csv(
        os.path.join(path, 'Error_rate_1-499.txt'), sep='\t', index_col=0
    )
    fig_error, ax_error = plt.subplots()
    ax_error.plot(df_k_error['k'], df_k_error['error_train'], '-b',
                  label='train')
    ax_error.plot(df_k_error['k'], df_k_error['error_test'], '--r',
                  label='test')
    leg = ax_error.legend()
    ax_error.set_xlabel('Numbers of dimensions')
    ax_error.set_ylabel('Error')
    fig_error.savefig(
        os.path.join(path, 'Error plot 1-499.png')
    )

    # eigenface algorithm
    # split data set
    info_trainset = df_info.loc[df_info['group'] == 'train', :]
    info_testset = df_info.loc[df_info['group'] == 'test', :]
    df_trainset = df_images.loc[info_trainset['filename'], :]
    df_testset = df_images.loc[info_testset['filename'], :]

    obj_pca_lda = EigenFaceAndFisher(df_trainset, info_trainset)
    pca_vec_project, red_pca_train = obj_pca_lda.get_eigen_projection(160)
    lda_vec_project, red_lda_train = obj_pca_lda.get_fisher_projection(160, 39)
    pca_lda_labels = obj_pca_lda.predict_label(
        df_testset, pca_vec_project, lda_vec_project
    )
    acc = accuracy_score(info_testset['person_id'], pca_lda_labels)
    print("Accuracy of PCA+LDA: ", acc)

    time_end = time()
    print(time_end - time_start)
