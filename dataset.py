# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:56:56 2017

@author: yamane
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import utility

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
                frame_rgb = utility.crop_108(frame_rgb)
                hist_gray = utility.crop_108(hist_gray)
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
        for l in open(r'E:\50Salads\eval_classes.txt').readlines():
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
            class_label = '_'.join(data[2].split('_')[:-1])
            if data[2] == word1 or data[2] == word2 or data[2] == word3:
                continue
            if class_label == 'serve_salad_onto_plate' or class_label == 'mix_dressing' or class_label == 'mix_ingredients' or class_label == 'add_dressing':
                data[2] = class_label
            elif class_label == 'peel_cucumber':
                data[2] = 'peel'
            elif class_label == 'cut_cucumber' or class_label == 'cut_tomato' or class_label == 'cut_cheese' or class_label == 'cut_lettuce':
                data[2] = 'cut'
            elif class_label == 'place_cucumber_into_bowl' or class_label == 'place_tomato_into_bowl' or class_label == 'place_cheese_into_bowl' or class_label == 'place_lettuce_into_bowl':
                data[2] = 'place'
            elif class_label == 'add_oil' or class_label == 'add_vinegar':
                data[2] = 'add_oil'
            elif class_label == 'add_salt' or class_label == 'add_pepper':
                data[2] = 'add_pepper'
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
            data = [time.index(anno[i][0]), time.index(anno[i][1]), self.class_uniq.index(anno[i][2])]
            sec.append(data)
        return sec

    def create_target_list(self, time_list, sec):
        targets = [0] * len(time_list)
        for c in range(len(sec)):
            for f in range(sec[c][0], sec[c][1]):
                targets[f] = sec[c][2]
        return targets

    def full_video(self):
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
                frame_rgb = utility.crop_108(frame_rgb)
                hist_gray = utility.crop_108(hist_gray)
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
        return batches, targets, self.finish

if __name__ == '__main__':
    start = time.time()
    dataset_root_dir = r'E:\50Salads\rgb'
    annotation_dir = r'E:\50Salads\ann-ts'
    timestamp_dir = r'E:\50Salads\time_stamp'
    video_pathes = utility.create_path_list(dataset_root_dir)
    anno_pathes = utility.create_path_list(annotation_dir)
    time_pathes = utility.create_path_list(timestamp_dir)
    permu = np.random.permutation(len(video_pathes))
    video_pathes = utility.list_shuffule(video_pathes, permu)
    anno_pathes = utility.list_shuffule(anno_pathes, permu)
    time_pathes = utility.list_shuffule(time_pathes, permu)
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
