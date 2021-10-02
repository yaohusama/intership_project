import re
import os
import csv
import sys
import numpy as np
import pickle as pkl
from random import sample
from DataProcess.config import ConfigParam


class DataParse:

    def random_embedding(self, embedding_dim, word_num):
        """
        随机的生成word的embedding，这里如果有语料充足的话，可以直接使用word2vec蓄念出词向量，这样词之间的区别可能更大。
        :param embedding_dim:  词向量的维度。
        :return: numpy format array. shape is : (vocab, embedding_dim)
        """
        # 根据字典大小word_num给字典内每一个单词生成一个embedding_dim维度的向量。
        # 采用截断正太分布的方式生成。前提是假设你的数据集满足正太分布
        embedding_mat = np.random.uniform(-0.25, 0.25, (word_num, embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat


class Data_Inter:
    """
    生成训练数据
    """
    def __init__(self, vocab_au, vocab_key):
        self.config = ConfigParam()
        self.vocab_au = vocab_au  # 字典路径
        self.vocab_key = vocab_key  # 字典路径
        self.index = 0
        self.index_test = 0
        self.reload_num = 10
        # train set
        self.task_sentence = pkl.load(open('DataProcess/data_manage_key_mapping.pkl', mode='rb'))
        self.au_all, self.ke_all = self.get_author_art()
        print(self.au_all.shape, self.ke_all.shape)
        # TRAIN
        self.get_data_mapping()

    def get_data_mapping(self):
        self.shuffle_all = sample(range(0, self.au_all.shape[0], 1), self.au_all.shape[0])
        self.all_sam = len(self.shuffle_all)
        self.au = self.au_all[self.shuffle_all[: int(self.all_sam * 0.7)]]  # 0.8
        self.ke = self.ke_all[self.shuffle_all[: int(self.all_sam * 0.7)]]
        self.au_test = self.au_all[self.shuffle_all[int(self.all_sam * 0.7):]]
        self.ke_test = self.ke_all[self.shuffle_all[int(self.all_sam * 0.7):]]
        self.shuffle = sample(range(0, self.au.shape[0], 1), self.au.shape[0])
        self.shuffle_test = sample(range(0, self.au_test.shape[0], 1), self.au_test.shape[0])
        self.end = self.au.shape[0]
        self.end_test = self.au_test.shape[0]

    def get_author_art(self):
        auth = []
        keys = []
        for j in self.task_sentence:
            for k in j:
                auth.append(k[: 2])
                keys.append(k[2][: 7])
        return np.array(auth), np.array(keys)

    def next(self):
        # 获取批
        sentence = []
        task_ = []
        if self.index + self.config.batch_size < self.end:  # 没有遍历到末尾
            it_data = self.shuffle[self.index: self.index + self.config.batch_size]
            self.index += self.config.batch_size  # 标识获取批次的索引值随批改变
        elif self.index + self.config.batch_size == self.end:  # 刚好遍历到末尾
            it_data = self.shuffle[self.index + self.config.batch_size: self.end]
            self.shuffle = sample(range(0, self.end, 1), self.end)
            self.index = 0
        else:
            it_data = self.shuffle[self.index: self.end]  # 随机选取
            self.shuffle = sample(range(0, self.end, 1), self.end)
            remain = self.shuffle[: self.index + self.config.batch_size - self.end]  # 剩余
            it_data = np.concatenate((it_data, remain), axis=0)
            self.index = 0
            if self.reload_num > 0:
                self.get_data_mapping()
                self.reload_num -= 1
        # print('it_data:', it_data)
        sentences_au = self.au[it_data]
        sentences_key = self.ke[it_data]
        for cur_sentences, cur_task in zip(sentences_au, sentences_key):
            task_.append(self.sentence2index(cur_sentences, self.vocab_au))  # author mapping
            sentence.append(self.sentence2index(cur_task, self.vocab_key))  # keys mapping
        return np.array(sentence), np.array(task_)

    def next_test(self):
        # 获取批
        sentence = []
        task_ = []
        if self.index_test + self.config.batch_size <= self.end_test:  # 没有遍历到末尾
            # 从task_sentence里面获取self.config.batch_size大小的块
            it_data = self.shuffle_test[self.index_test: self.index_test + self.config.batch_size]  # 迭代数据
            self.index_test += self.config.batch_size  # 标识获取批次的索引值随批改变
        elif self.index_test + self.config.batch_size == self.end_test:  # 刚好遍历到末尾
            it_data = self.shuffle_test[self.index_test + self.config.batch_size: self.end_test]
            self.shuffle_test = sample(range(0, self.end_test, 1), self.end_test)
            self.index_test = 0  # 重置index。非常简单的小学数学问题，后面不在注释了，一眼便懂
        else:
            it_data = self.shuffle_test[self.index_test: self.end_test]  # 随机选取
            self.shuffle_test = sample(range(0, self.end_test, 1), self.end_test)
            remain = self.shuffle_test[: self.index_test + self.config.batch_size - self.end_test]  # 剩余
            it_data = np.concatenate((it_data, remain), axis=0)
            self.index_test = 0
        sentences_au = self.au_test[it_data]
        sentences_key = self.ke_test[it_data]
        for cur_sentences, cur_task in zip(sentences_au, sentences_key):
            task_.append(self.sentence2index(cur_sentences, self.vocab_au))  # author mapping
            sentence.append(self.sentence2index(cur_task, self.vocab_key))  # keys mapping
        return np.array(sentence), np.array(task_)

    def sentence2index(self, sen, vocab):
        sen2id = []
        for cur_sen in sen:
            sen2id.append(vocab.get(cur_sen, 0))  # 如果找不到，就用0代替。
        return sen2id

    def task2index(self, cur_tasks, mapping):
        assert isinstance(cur_tasks, list) and len(cur_tasks) > 0 and hasattr(cur_tasks, '__len__')
        assert isinstance(mapping, dict) and len(mapping) > 0
        cur_task2index_mapping = []
        for cur_task in cur_tasks:
            cur_task2index_mapping.append(mapping[cur_task])
        return cur_task2index_mapping


if __name__ == '__main__':
    from DataProcess.dataset_info import info
    au_vocab, key_vocab, au_len, key = info('DataProcess/data_manage_key_mapping.pkl')
    so = Data_Inter(au_vocab, key_vocab)
