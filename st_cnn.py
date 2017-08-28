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

import cupy as cp

import chainer
from chainer import cuda, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import utility
from dataset import Dataset
from links import CBR

class SpatialConv(chainer.Chain):
    def __init__(self):
        super(SpatialConv, self).__init__(
            cbr1=CBR(4, 32, 3, stride=1, pad=1),
            cbr2= CBR(32, 64, 3, stride=1, pad=1),
            cbr3= CBR(64, 96, 3, stride=1, pad=1),
            cbr4= CBR(96, 128, 3, stride=1, pad=1),
            fc1=L.Linear(None, 256),
            fc2=L.Linear(256, 128)
        )

    def __call__(self, x_btchw):
        """
        Args:
            shape = (b, t, c, h, w).
        Returns:
            shape = (b * t, d).
        """
        x_tchw = x_btchw.reshape(x_btchw.shape[0]*x_btchw.shape[1],
                                 x_btchw.shape[2],
                                 x_btchw.shape[3],
                                 x_btchw.shape[4])
        h = self.cbr1(x_tchw)
        h = F.max_pooling_2d(h, 3)
        h = self.cbr2(h)
        h = F.max_pooling_2d(h, 3)
        h = self.cbr3(h)
        h = F.max_pooling_2d(h, 3)
        h = self.cbr4(h)
        h = F.max_pooling_2d(h, 3)
        h = F.relu(self.fc1(h))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))
        y= F.dropout(h)
        return y

class TemporalConv(chainer.Chain):
    def __init__(self):
        super(TemporalConv, self).__init__(
            conv=L.ConvolutionND(1, 128, 10, 41, pad=20),
        )

    def __call__(self, x_bdt):
        """
        Args:
            shape = (b, d, t).
        Returns:
            shape = (b, k, t).
        """
        y = self.conv(x_bdt)
        return y

class STConv(chainer.Chain):
    def __init__(self):
        super(STConv, self).__init__(
            spatial=SpatialConv(),
            temporal=TemporalConv()
        )

    def __call__(self, x_btchw):
        """
        Args:
            shape = (b, t, c, h, w).
        Returns:
            shape = (b, k, t).
        """
        h_td = self.spatial(x_btchw)
        h_btd = h_td.reshape(x_btchw.shape[0], x_btchw.shape[1], h_td.shape[1])
        h_bdt = self.xp.transpose(h_btd, (0, 2, 1))
        y = self.temporal(h_bdt)
        y.data = self.xp.ascontiguousarray(y.data)
        return y

    def lossfun(self, x, t):
        y = self(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def loss_ave(self, creator):
        losses = []
        accuracies = []
        while(True):
            data = next(creator)
            x, t, finish = data
            x_btchw = x.reshape(1, 600, 4, 108, 108)
            t_bt = t.reshape(1, 600)
            x_btchw = x_btchw.astype('f')
            x_btchw = cuda.to_gpu(x_btchw)
            t_bt = cuda.to_gpu(t_bt)
            # 順伝播を計算し、誤差と精度を取得
            with chainer.using_config('train', False):
                loss, accuracy = self.lossfun(x_btchw, t_bt)
            # 逆伝搬を計算
            losses.append(cuda.to_cpu(loss.data))
            accuracies.append(cuda.to_cpu(accuracy.data))
            if finish is True:
                break
        return np.mean(losses), np.mean(accuracies)

    def predict(self, x):
        """
        Args:
            shape = (b, t, c, h, w).
        Returns:
            shape = (b, t, k).
        """
        with chainer.using_config('train', False):
            y = self(x)
        y_bkt = F.softmax(y)
        y_btk = self.xp.transpose(y_bkt, (0, 2, 1))
        return y_btk


if __name__ == '__main__':
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 超パラメータ
    max_iteration = 1500000  # 繰り返し回数
    batch_size = 600
    num_train = 40  # 学習データ数
    num_valid = 5  # 検証データ数
    learning_rate = 0.00003  # 学習率
    # 初期設定
    time_start = time.time()
    epoch_loss = []
    epoch_valid_loss = []
    epoch_accuracy = []
    epoch_valid_accuracy = []
    loss_valid_best = np.inf
    accuracy_valid_best = 0.0
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
    video_root_dir = r'E:\50Salads\rgb'
    anno_root_dir = r'E:\50Salads\ann-ts'
    time_root_dir = r'E:\50Salads\time_stamp'
    video_pathes = utility.create_path_list(video_root_dir)
    anno_pathes = utility.create_path_list(anno_root_dir)
    time_pathes = utility.create_path_list(time_root_dir)
    # イテレータを作成
    train_data = Dataset(batch_size, video_pathes, anno_pathes, time_pathes, 0, 30)
    valid_data = Dataset(batch_size, video_pathes, anno_pathes, time_pathes, 30, 45)
    test_data = Dataset(batch_size, video_pathes, anno_pathes, time_pathes, 45, 50)
    # モデル読み込み
    model = STConv().to_gpu()
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
                x_btchw = x.reshape(1, 600, 4, 108, 108)
                t_bt = t.reshape(1, 600)
                x_btchw = x_btchw.astype('f')
                x_btchw = cuda.to_gpu(x_btchw)
                t_bt = cuda.to_gpu(t_bt)
                # 順伝播を計算し、誤差と精度を取得
                with chainer.using_config('train', False):
                    loss, accuracy = model.lossfun(x_btchw, t_bt)
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
                best_epoch = epoch
                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print()
            print("epoch:", epoch)
            print("time", epoch_time, "(", total_time, ")")
            print("loss[train]:", epoch_loss[epoch])
            print("loss[valid]:", loss_valid)
            print("loss[valid_best]:", loss_valid_best)
            print()
            print("accuracy[train]:", epoch_accuracy[epoch])
            print("accuracy[valid]:", accuracy_valid)
            print("accuracy[valid_best]:", accuracy_valid_best)
            print()
            print("best epoch:", best_epoch)

            x, t, finish = next(test_data)
            x_btchw = x.reshape(1, 600, 4, 108, 108)
            x_btchw = x_btchw.astype('f')
            x_btchw = cuda.to_gpu(x_btchw)
            y = model_best.predict(x_btchw)
            print()
            print('predict:', test_data.class_uniq[int(cp.argmax(y.data[0]))])
            print('target:', test_data.class_uniq[t[0]])
            plt.subplot(121)
            plt.imshow(np.transpose(cuda.to_cpu(x[0]), (1, 2, 0))[:, :, :3])
            plt.subplot(122)
            plt.imshow(np.transpose(cuda.to_cpu(x[0]), (1, 2, 0))[:, :, 3])
            plt.gray()
            plt.show()

            if (epoch % 10) == 0:
                plt.plot(epoch_loss)
                plt.plot(epoch_valid_loss)
                plt.title("loss")
                plt.legend(["train", "valid"], loc="upper right")
                plt.grid()
                plt.show()

                plt.plot(epoch_accuracy)
                plt.plot(epoch_valid_accuracy)
                plt.title("accuracy")
                plt.legend(["train", "valid"], loc="lower right")
                plt.grid()
                plt.show()

                while(True):
                    x, t, finish = next(test_data)
                    x_btchw = x.reshape(1, 600, 4, 108, 108)
                    x_btchw = x_btchw.astype('f')
                    x_btchw = cuda.to_gpu(x_btchw)
                    y = model_best.predict(x_btchw)
                    print('y')
                    utility.plot_bar(y)
                    print('t')
                    utility.plot_bar(t)
                    if finish is True:
                        break

    except KeyboardInterrupt:
        print("割り込み停止が実行されました")

    plt.plot(epoch_loss)
    plt.plot(epoch_valid_loss)
    plt.title("loss")
    plt.legend(["train", "valid"], loc="upper right")
    plt.grid()
    plt.savefig(loss_filename)
    plt.show()

    plt.plot(epoch_accuracy)
    plt.plot(epoch_valid_accuracy)
    plt.title("accuracy")
    plt.legend(["train", "valid"], loc="lower right")
    plt.grid()
    plt.savefig(accuracy_filename)
    plt.show()

#    while(True):
#        x, t, finish = test_data.full_video()
#        x = x.astype('f')
#        x = cuda.to_gpu(x)
#        y = model_best.predict(x)
#        print('y', test_data.class_uniq[int(cp.argmax(y.data[0]))])
#        print('t', test_data.class_uniq[t[0]])
#        plt.subplot(121)
#        plt.imshow(np.transpose(cuda.to_cpu(x[0]), (1, 2, 0))[:, :, :3])
#        plt.subplot(122)
#        plt.imshow(np.transpose(cuda.to_cpu(x[0]), (1, 2, 0))[:, :, 3])
#        plt.gray()
#        plt.show()
#        print('y')
#        utility.plot_bar(y)
#        print('t')
#        utility.plot_bar(t)
#        if finish is True:
#            break
    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    print('num_iteration:', epoch)
    print('best_epoch:', best_epoch)
    print('learning_rate:', learning_rate)
    print('batch_size:', batch_size)