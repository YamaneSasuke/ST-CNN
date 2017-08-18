# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:45:30 2017

@author: yamane
"""

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
        elif y.shape[1] == 18:
            li = []
            for f in range(len(y.data)):
                for i in range(len(y.data[f])):
                    if y.data[f][i] == max(y.data[f]):
                        max_index = i
                li.append(max_index)
            y = np.stack(li)
    num_features = len(y)
    num_index = 18
    colors=plt.cm.jet(np.linspace(0,1,num_index))
    for i in range(num_features):
        plt.axvspan(i, i+1, color=colors[int(y[i])])
    plt.xlim(0, num_features)
    plt.show()
