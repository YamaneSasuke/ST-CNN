# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:45:30 2017

@author: yamane
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_bar(y):
    """
    y:
    type = ndarray, shape = (num_feature, index)
    or (num_features,)
    or (num_features, 18)
    """
    if len(y.shape) == 2:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        elif y.shape[1] == 10:
            li = []
            for f in range(len(y.data)):
                for i in range(len(y.data[f])):
                    if y.data[f][i] == max(y.data[f]):
                        max_index = i
                li.append(max_index)
            y = np.stack(li)
    num_features = len(y)
    num_index = 10
    colors=plt.cm.jet(np.linspace(0,1,num_index))
    for i in range(num_features):
        plt.axvspan(i, i+1, color=colors[int(y[i])])
    plt.xlim(0, num_features)
    plt.show()

def create_path_list(dataset_root_dir):
    path_list = []
    for root, dirs, files in os.walk(dataset_root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            path_list.append(file_path)
    return path_list

def list_shuffule(ori_list, permu):
    l = []
    for i in permu:
        l.append(ori_list[i])
    return l

def crop_108(image, top, left, flip):
    h_image, w_image = image.shape[:2]
    h_crop = 108
    w_crop = 108
    bottom = top + h_crop
    right = left + w_crop
    image = image[top:bottom, left:right]
    if bool(flip) is True:
        image = image[:, ::-1]  # 左右反転
    return image

def random_crop_and_flip(image, crop_size):
    h_image, w_image = image.shape[:2]
    h_crop = crop_size
    w_crop = crop_size
    # 0以上 h_image - h_crop以下の整数乱数
    top = np.random.randint(0, h_image - h_crop + 1)
    left = np.random.randint(0, w_image - w_crop + 1)
    bottom = top + h_crop
    right = left + w_crop
    image = image[top:bottom, left:right]

    if np.random.rand() > 0.5:  # 半々の確率で
        image = image[:, ::-1]  # 左右反転

    return image