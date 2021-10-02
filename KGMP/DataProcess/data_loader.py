# coding=utf-8
import os
import sys
import numpy as np
import pickle as pkl
from random import sample
from random import randint
import jieba.posseg as pseg
from DataProcess.utils import load_mnist
from DataProcess.config import ConfigParam


class DataLoad:
    def __init__(self, rate, flag):
        self.config = ConfigParam()
        self.rate = rate
        self.label_train, self.record_train, self.label_test, self.record_test = self.create_data(rate=self.rate)
        print('cur_trainer:', flag, "private data info below:\t")
        print('\ttrain scare:', str(self.record_train.shape), '\t', 'test scare:', str(self.record_test.shape))
        # ######################################################################
        self.end_record = self.record_train.shape[0]
        self.index = 0
        self.batchs = self.end_record // self.config.batch_size
        self.shuffle_record = sample(range(0, self.end_record, 1), self.end_record)
        # test
        self.end_record_test = self.record_test.shape[0]
        self.index_test = 0
        self.batchs_test = self.end_record_test // self.config.batch_size
        self.shuffle_record_test = sample(range(0, self.end_record_test, 1), self.end_record_test)

    def create_data(self, rate=[0., 1]):
        trX, trY, valX, valY = load_mnist()
        return trY[int(trX.shape[0] * rate[0]): int(trX.shape[0] * rate[1])], \
               trX[int(trX.shape[0] * rate[0]): int(trX.shape[0] * rate[1])], \
               valY[int(valX.shape[0] * rate[0]): int(valX.shape[0] * rate[1])], \
               valX[int(valX.shape[0] * rate[0]): int(valX.shape[0] * rate[1])],

    def next(self):
        # 获取批
        if self.index + self.config.batch_size == self.end_record:
            self.index = self.config.batch_size
            self.shuffle_record = sample(range(0, self.end_record, 1), self.end_record)
            cur_index_selected = self.shuffle_record[0: self.index]  # 迭代索引
        elif self.index + self.config.batch_size < self.end_record:
            cur_index_selected = self.shuffle_record[self.index: self.index + self.config.batch_size]  # 迭代索引
            self.index = self.index + self.config.batch_size
        else:
            cur_index_selected = self.shuffle_record[self.index: self.end_record]  # 28~31  35 - 31 = 4
            self.index = self.index + self.config.batch_size - self.end_record
            self.shuffle_record = sample(range(0, self.end_record, 1), self.end_record)
            remain_index = self.shuffle_record[0: self.index]
            cur_index_selected.extend(remain_index)
        task = self.label_train[cur_index_selected]
        data = self.record_train[cur_index_selected]
        return data, task

    def next_test(self):
        # 获取批
        if self.index_test + self.config.batch_size == self.end_record_test:
            self.index_test = self.config.batch_size
            self.shuffle_record_test = sample(range(0, self.end_record_test, 1), self.end_record_test)
            cur_index_selected = self.shuffle_record_test[0: self.index_test]  # 迭代索引
        elif self.index_test + self.config.batch_size < self.end_record_test:
            cur_index_selected = self.shuffle_record_test[self.index_test: self.index_test + self.config.batch_size]  # 迭代索引
            self.index_test = self.index_test + self.config.batch_size
        else:
            cur_index_selected = self.shuffle_record_test[self.index_test: self.end_record_test]  # 28~31  35 - 31 = 4
            self.index_test = self.index_test + self.config.batch_size - self.end_record_test
            self.shuffle_record_test = sample(range(0, self.end_record_test, 1), self.end_record_test)
            remain_index = self.shuffle_record_test[0: self.index_test]
            cur_index_selected.extend(remain_index)
        task = self.label_test[cur_index_selected]
        data = self.record_test[cur_index_selected]
        return data, task
        # return np.array(sentence), np.array(task)


if __name__ == "__main__":
    dataLoad = DataLoad(rate=[0.2, 0.5])
    # dataLoad.create_data()
    batches_recording = 0
    # while batches_recording <= dataLoad.batchs:  #
    while batches_recording <= 0:  #
        batches_recording += 1
        # d, l, t, p = dataLoad.next_batch()
        d, l = dataLoad.next()
        print('data info:', l.shape, len(d))  # (21, 21, 160, 300) (21,) (21, 21, 160) (21, 21, 160)
        print(d[0])  # (21, 21, 160, 300) (21,) (21, 21, 160) (21, 21, 160)
#     data = pkl.load(open('../data/News/news_data_embedding.pkl', mode='rb'))
    # data_vocab = pkl.load(open('vocab', mode='rb'))
    # print(data)
    # print(data_vocab['<PAD>'])
