# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:18:11 2017

@author: yamane
"""

import os
import time
import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, optimizers, serializers
import chainer.functions as F
import chainer.links as L

from dataset import Dataset
import dataset


class SpatialNet(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(SpatialNet, self).__init__(
            conv1=L.Convolution2D(6, 64, 3, stride=1, pad=1),
            conv2=L.Convolution2D(64, 96, 3, stride=1, pad=1),
            conv3=L.Convolution2D(96, 128, 3, stride=1, pad=1),

            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(96),
            bn3=L.BatchNormalization(128),

            fc4=L.Linear(3200, 256),
            fc5=L.Linear(256, 18),
        )
        self.train = False

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pooling_2d(h, 3)
        h = self.fc4(h)
        y = self.fc5(h)
        return y

    def forward(self, x):
        y = self(x)
        return y

    def lossfun(self, x, t):
        y = self.forward(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def loss_ave(self, creator):
        losses = []
        accuracies = []
        while(True):
            data = next(creator)
            x, t, finish = data
            x = x.astype('f')
            x = cuda.to_gpu(x)
            t = cuda.to_gpu(t)
            # 順伝播を計算し、誤差と精度を取得
            with chainer.using_config('train', False):
                loss, accuracy = self.lossfun(x, t)
            # 逆伝搬を計算
            losses.append(cuda.to_cpu(loss.data))
            accuracies.append(cuda.to_cpu(accuracy.data))
            if finish is True:
                break
        return np.mean(losses), np.mean(accuracies)

    def predict(self, x):
        with chainer.using_config('train', False):
            y = self.forward(x)
        return F.softmax(y)


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    time_start = time.time()
    epoch_loss = []
    epoch_valid_loss = []
    epoch_accuracy = []
    epoch_valid_accuracy = []
    loss_valid_best = np.inf
    accuracy_valid_best = 0.0

    # 超パラメータ
    max_iteration = 1500000  # 繰り返し回数
    batch_size = 200
    num_train = 40  # 学習データ数
    num_valid = 5  # 検証データ数
    learning_rate = 0.00003  # 学習率

    video_root_dir = r'E:\50Salads\rgb'
    anno_root_dir = r'E:\50Salads\ann-ts'
    time_root_dir = r'E:\50Salads\time_stamp'
    video_pathes = dataset.create_path_list(video_root_dir)
    anno_pathes = dataset.create_path_list(anno_root_dir)
    time_pathes = dataset.create_path_list(time_root_dir)
    train_data = Dataset(batch_size, video_pathes, anno_pathes, time_pathes, 0, 40)
    valid_data = Dataset(batch_size, video_pathes, anno_pathes, time_pathes, 40, 45)
    test_data = Dataset(batch_size, video_pathes, anno_pathes, time_pathes, 45, 50)

    # 学習結果保存場所
    output_location = r'C:\Users\yamane\OneDrive\M1\SpatialNet'
    # 学習結果保存フォルダ作成
    output_root_dir = os.path.join(output_location, file_name)
    folder_name = str(time_start)
    output_root_dir = os.path.join(output_root_dir, folder_name)
    if os.path.exists(output_root_dir):
        pass
    else:
        os.makedirs(output_root_dir)
    # ファイル名を作成
    model_filename = str(file_name) + '.npz'
    loss_filename = 'epoch_loss' + str(time_start) + '.png'
    accuracy_filename = 'epoch_accuracy' + str(time_start) + '.png'
    model_filename = os.path.join(output_root_dir, model_filename)
    loss_filename = os.path.join(output_root_dir, loss_filename)
    accuracy_filename = os.path.join(output_root_dir, accuracy_filename)

    # モデル読み込み
    model = SpatialNet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    time_origin = time.time()
    try:
        for epoch in range(max_iteration):
            time_begin = time.time()
            losses = []
            accuracies = []
            for i in tqdm.tqdm(range(40)):
                data = next(train_data)
                x, t, finish = data
                x = x.astype('f')
                x = cuda.to_gpu(x)
                t = cuda.to_gpu(t)
                # 勾配を初期化
                model.cleargrads()
                # 順伝播を計算し、誤差と精度を取得
                with chainer.using_config('train', True):
                    loss, accuracy = model.lossfun(x, t)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))
                accuracies.append(cuda.to_cpu(accuracy.data))
                if finish is True:
                    break

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))
            epoch_accuracy.append(np.mean(accuracies))

            loss_valid, accuracy_valid = model.loss_ave(valid_data)
            epoch_valid_loss.append(loss_valid)
            epoch_valid_accuracy.append(accuracy_valid)
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                accuracy_valid_best = accuracy_valid
                epoch__loss_best = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print()
            print("epoch:", epoch)
            print("time", epoch_time, "(", total_time, ")")
            print("loss[train]:", epoch_loss[epoch])
            print("loss[valid]:", loss_valid)
            print("loss[valid_best]:", loss_valid_best)
            print("accuracy[train]:", epoch_accuracy[epoch])
            print("accuracy[valid]:", accuracy_valid)
            print("accuracy[valid_best]:", accuracy_valid_best)
            print("epoch[valid_best]:", epoch__loss_best)

            if (epoch % 10) == 0:
                plt.figure(figsize=(16, 12))
                plt.plot(epoch_loss)
                plt.plot(epoch_valid_loss)
                plt.title("loss")
                plt.legend(["train", "valid"], loc="upper right")
                plt.grid()
                plt.show()

                plt.figure(figsize=(16, 12))
                plt.plot(epoch_accuracy)
                plt.plot(epoch_valid_accuracy)
                plt.title("accuracy")
                plt.legend(["train", "valid"], loc="lower right")
                plt.grid()
                plt.show()

    except KeyboardInterrupt:
        print("割り込み停止が実行されました")

    plt.figure(figsize=(16, 12))
    plt.plot(epoch_loss)
    plt.plot(epoch_valid_loss)
    plt.title("loss")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    plt.figure(figsize=(16, 12))
    plt.plot(epoch_accuracy)
    plt.plot(epoch_valid_accuracy)
    plt.title("accuracy")
    plt.legend(["train", "valid"], loc="lower right")
    plt.grid()
    plt.savefig(accuracy_filename)
    plt.show()

    i = 0

    for data in test_data:
        x, t, finish = data
        x = x.astype('f')
        x = cuda.to_gpu(x)
        y = model_best.predict(x)
        print('test_num:', i)
        print('y', y.data[0])
        print('t', t[0])
        i += 1
        if finish is True:
            break
    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    print('max_iteration:', max_iteration)
    print('learning_rate:', learning_rate)
    print('train_size', 200)
    print('valid_size', 30)
    print('trim_size', 20)
