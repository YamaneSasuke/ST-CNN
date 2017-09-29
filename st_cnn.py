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
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers
from chainer.iterators import SerialIterator
from chainer.iterators import MultiprocessIterator

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

    def __call__(self, x_tchw):
        """
        Args:
            shape = (t, c, h, w).
        Returns:
            shape = (b * t, d).
        """
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
            conv=L.ConvolutionND(1, 128, 10, 41),
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

    def __call__(self, x_tchw):
        """
        Args:
            shape = (t, c, h, w).
        Returns:
            shape = (b, k, t).
        """
        h_td = self.spatial(x_tchw)
        h_btd = h_td.reshape(1, h_td.shape[0], h_td.shape[1])
        h_bdt = h_btd.transpose(0, 2, 1)
        y = self.temporal(h_bdt)
        y.data = self.xp.ascontiguousarray(y.data)
        return y

    def lossfun(self, x, t, class_weight=None):
        y = self(x)
        loss = F.softmax_cross_entropy(y, t, class_weight=class_weight)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def loss_ave(self, iterator):
        losses = []
        accuracies = []
        while(True):
            data = next(iterator)
            x_tchw = data[0][0]
            t = data[0][1]
            finish = data[0][2]
            t_bt = t.reshape(1, t.shape[0])
            x_tchw = x_tchw.astype('f')
            x_tchw = cuda.to_gpu(x_tchw)
            t_bt = cuda.to_gpu(t_bt)
            # 順伝播を計算し、誤差と精度を取得
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                loss, accuracy = self.lossfun(x_tchw, t_bt)
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
    __spec__ = None
    file_name = os.path.splitext(os.path.basename(__file__))[0]

    # 超パラメータ
    max_iteration = 1500000  # 繰り返し回数
    batch_size = 1
    num_frame = 600
    num_train_video = 30  # 学習データ数
    num_valid_video = 15  # 検証データ数
    num_test_video = 5  # 検証データ数
    learning_rate = 0.00003  # 学習率
    # 学習結果保存場所
    output_location = r'C:\Users\yamane\OneDrive\M1\SpatialNet'
    # 学習データの保存場所
    video_root_dir = r'E:\50Salads\rgb'
    anno_root_dir = r'E:\50Salads\ann-ts'
    time_root_dir = r'E:\50Salads\time_stamp'

    # 初期設定
    time_start = time.time()
    epoch_loss = []
    epoch_valid_loss = []
    epoch_accuracy = []
    epoch_valid_accuracy = []
    loss_valid_best = np.inf
    accuracy_valid_best = 0.0
    best_epoch = 0
    # 学習データのパスのリストを作成
    video_pathes = utility.create_path_list(video_root_dir)
    anno_pathes = utility.create_path_list(anno_root_dir)
    time_pathes = utility.create_path_list(time_root_dir)
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

    # イテレータを作成
    train_data = Dataset(num_frame, video_pathes, anno_pathes, time_pathes,
                         0, num_train_video)
    valid_data = Dataset(num_frame, video_pathes, anno_pathes, time_pathes,
                         num_train_video, num_train_video+num_valid_video)
    test_data = Dataset(num_frame, video_pathes, anno_pathes, time_pathes,
                        num_train_video+num_valid_video, 50)
#    train_ite = SerialIterator(train_data, 1)
#    valid_ite = SerialIterator(valid_data, 1)
#    test_ite = SerialIterator(test_data, 1)
    train_ite = MultiprocessIterator(train_data, 1, n_processes=2)
    valid_ite = MultiprocessIterator(valid_data, 1, n_processes=2)
    test_ite = MultiprocessIterator(test_data, 1, n_processes=2)
    # class_weightを設定
    dis_list = train_data.target_distribution()
    cw = cp.ndarray((10,), 'f')
    for i in range(10):
        cw[i] = 10 / dis_list[i]
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
            for i in tqdm.tqdm(range(num_train_video)):
                data = next(train_ite)
                x = data[0][0]
                t = data[0][1]
                finish = data[0][2]
                t_bt = t.reshape(batch_size, num_frame)
                x_tchw = x.astype('f')
                x_tchw = cuda.to_gpu(x_tchw)
                t_bt = cuda.to_gpu(t_bt)
                # 勾配を初期化
                model.cleargrads()
                # 順伝播を計算し、誤差と精度を取得
                with chainer.using_config('train', True):
                    loss, accuracy = model.lossfun(x_tchw, t_bt, cw)
                    # 逆伝搬を計算
                    loss.backward()
                optimizer.update()
                losses.append(cuda.to_cpu(loss.data))
                accuracies.append(cuda.to_cpu(accuracy.data))
                if finish is True:
                    break
            # 訓練データの結果を保持
            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            epoch_loss.append(np.mean(losses))
            epoch_accuracy.append(np.mean(accuracies))
            # 検証データの結果を保持
            loss_valid, accuracy_valid = model.loss_ave(valid_ite)
            epoch_valid_loss.append(loss_valid)
            epoch_valid_accuracy.append(accuracy_valid)
            if loss_valid < loss_valid_best:
                loss_valid_best = loss_valid
                accuracy_valid_best = accuracy_valid
                best_epoch = epoch
                model_best = copy.deepcopy(model)
            # 訓練データ,検証データの結果を表示
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
            # テストデータの予測結果を表示
            data = next(test_ite)
            x_tchw = data[0][0]
            t = data[0][1]
            finish = data[0][2]
            x_tchw = x_tchw.astype('f')
            x_tchw = cuda.to_gpu(x_tchw)
            y = model_best.predict(x_tchw)
            print()
            print('predict:', test_data.class_uniq[int(cp.argmax(y.data[0][0]))])
            print('target:', test_data.class_uniq[t[0]])
            plt.subplot(121)
            plt.imshow(np.transpose(cuda.to_cpu(x[0]), (1, 2, 0))[:, :, :3])
            plt.subplot(122)
            plt.imshow(np.transpose(cuda.to_cpu(x[0]), (1, 2, 0))[:, :, 3])
            plt.gray()
            plt.show()
            # 10エポックごとに学習推移をグラフで表示
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
                    data = next(test_ite)
                    x_tchw = data[0][0]
                    t = data[0][1]
                    finish = data[0][2]
                    x_tchw = x_tchw.astype('f')
                    x_tchw = cuda.to_gpu(x_tchw)
                    y = model_best.predict(x_tchw)
                    print('y')
                    utility.plot_bar(y[0])
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
    # ベストモデルを保存
    model_filename = os.path.join(output_root_dir, model_filename)
    serializers.save_npz(model_filename, model_best)

    print('num_iteration:', epoch)
    print('best_epoch:', best_epoch)
    print('learning_rate:', learning_rate)
    print('batch_size:', batch_size)
    print('num_frame:', num_frame)
