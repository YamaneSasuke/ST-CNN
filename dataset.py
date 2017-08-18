# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:56:56 2017

@author: yamane
"""

import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Dataset(object):
    def __init__(self, batch_size, video_pathes, anno_pathes, time_pathes, start, end):
        self.batch_size = batch_size
        self.start = start
        self.end = end
        self.video_pathes = video_pathes[start:end]
        self.anno_pathes = anno_pathes[start:end]
        self.time_pathes = time_pathes[start:end]
        self.video_num = len(self.video_pathes)
        self.i = 0
        self.class_uniq = self.create_class_uniq()
        self.finish = False

    def __iter__(self):
        return self

    def __next__(self):
        self.finish = False
        i = 0
        # モーションの残存期間(sec)
        DURATION = 2.0
        batches = []
        targets = []

        anno_list = self.create_anno_list(self.anno_pathes[self.i])
        time_list = self.create_time_list(self.time_pathes[self.i])
        sec = self.create_sec_list(anno_list, time_list)
        target_list = self.create_target_list(time_list, sec)

        # ビデオのフレーム数を取得
        cap = cv2.VideoCapture(self.video_pathes[self.i])
        # フレームレートを取得
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        quater_frame_rate = int(frame_rate / 4)
        ret, frame = cap.read()
        frame_pre = frame.copy()
        # motion_historyの初期値
        height, width, channels = frame.shape
        motion_history = np.zeros((height, width), np.float32)
        while(ret):
            # フレーム間の差分計算
            color_diff = cv2.absdiff(frame, frame_pre)
            # グレースケール変換
            gray_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
            # ２値化
            retval, black_diff = cv2.threshold(gray_diff, 30, 1, cv2.THRESH_BINARY)
            # プロセッサ処理時間(sec)を取得
            proc_time = time.clock()
            # モーション履歴画像の更新
            cv2.motempl.updateMotionHistory(black_diff, motion_history, proc_time, DURATION)
            if (i % quater_frame_rate) == 0:
                # 古いモーションの表示を経過時間に応じて薄くする
                hist_color = np.array(np.clip((motion_history - (proc_time - DURATION)) / DURATION, 0, 1) * 255, np.uint8)
                # グレースケール変換
    #                hist_gray = cv2.cvtColor(hist_color, cv2.COLOR_GRAY2BGR)
                hist_gray = hist_color
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (128, 128))
                hist_gray = cv2.resize(hist_gray, (128, 128))
                frame_rgb = crop_108(frame_rgb)
                hist_gray = crop_108(hist_gray)
                hist_gray = hist_gray.reshape(108, 108, 1)
                image = np.concatenate((frame_rgb, hist_gray), axis=2)
                batches.append(image)
                targets.append(target_list[i])

            # 次のフレームの読み込み
            frame_pre = frame.copy()
            ret, frame = cap.read()
            i += 1

        # 終了処理
        cv2.destroyAllWindows()
        cap.release()

        batches = np.stack(batches, axis=0)
        targets = np.stack(targets, axis=0)
        batches = np.transpose(batches, (0, 3, 1, 2))
        batches = batches / 255.0
        targets = targets.astype(np.int32)

        self.i += 1
        if self.i == self.video_num:
            self.i = 0
            self.finish = True
        index = np.random.randint(0, len(batches)- self.batch_size)
        return batches[index: index + self.batch_size], targets[index: index + self.batch_size], self.finish

    def create_class_uniq(self):
        class_uniq = []
        for l in open(r'E:\50Salads\mid_classes.txt').readlines():
            data = l[:-1]
            class_uniq.append(data)
        return class_uniq

    def create_anno_list(self, anno_path):
        word1 = 'cut_and_mix_ingredients'
        word2 = 'prepare_dressing'
        word3 = 'serve_salad'
        anno_pathes = []
        for l in open(anno_path).readlines():
            data = l[:-1].split(' ')
            if data[2] == word1 or data[2] == word2 or data[2] == word3:
                continue
            anno_pathes.append(data)
        return anno_pathes

    def create_time_list(self, time_path):
        time_pathes = []
        for l in open(time_path).readlines():
            data = l[:-1].split(' ')[0]
            time_pathes.append(data)
        return time_pathes

    def create_sec_list(self, anno, time):
        sec = []
        for i in range(len(anno)):
            data = [time.index(anno[i][0]), time.index(anno[i][1]), self.class_uniq.index('_'.join(anno[i][2].split('_')[:-1]))]
            sec.append(data)
        return sec

    def create_target_list(self, time_list, sec):
        targets = [0] * len(time_list)
        for c in range(len(sec)):
            for f in range(sec[c][0], sec[c][1]):
                targets[f] = sec[c][2]
        return targets


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

def crop_108(image):
    h_image, w_image = image.shape[:2]
    h_crop = 108
    w_crop = 108
    top = int((h_image - h_crop )/ 2)
    left = int((w_image - w_crop) / 2)
    bottom = top + h_crop
    right = left + w_crop
    image = image[top:bottom, left:right]
    return image

if __name__ == '__main__':
    start = time.time()
    dataset_root_dir = r'E:\50Salads\rgb'
    annotation_dir = r'E:\50Salads\ann-ts'
    timestamp_dir = r'E:\50Salads\time_stamp'
    video_pathes = create_path_list(dataset_root_dir)
    anno_pathes = create_path_list(annotation_dir)
    time_pathes = create_path_list(timestamp_dir)
    permu = np.random.permutation(len(video_pathes))
    video_pathes = list_shuffule(video_pathes, permu)
    anno_pathes = list_shuffule(anno_pathes, permu)
    time_pathes = list_shuffule(time_pathes, permu)
    train = Dataset(600, video_pathes, anno_pathes, time_pathes, 0, 40)
#    valid = Dataset(200, video_pathes, anno_pathes, time_pathes, 40, 45)
    t_hist_t = []
#    t_hist_v = []
    batch_t, target_t, finish_t = next(train)
    for i in range(len(batch_t)):
        print(train.class_uniq[target_t[i]])
        plt.subplot(121)
        plt.imshow(np.transpose(batch_t[i], (1, 2, 0))[:, :, :3])
        plt.subplot(122)
        plt.imshow(np.transpose(batch_t[i], (1, 2, 0))[:, :, 3])
        plt.gray()
        plt.show()
