import csv
import jieba
import tensorflow as tf
from bi_lstm_model import Bi_LSTM_Task
# from bi_lstm_model_att import Bi_LSTM_Task
from DataProcess.dataset_info import info
from data_parse import DataParse, Data_Inter


class Config:
    def __init__(self):
        # self.root_data_path = 'data_path'
        self.root_data_path = r'model_save'
        self.batch_size = 32  # 批大小
        self.epoch = 1000000  # 训练周期，数据集被轮一遍的次数
        self.hidden_dim = 100  # lstm隐含层的节点个数
        self.optimizer = 'Adam'  # 优化函数
        self.lr = 0.001  # 学习率
        self.clip = 5.0  # 防止梯度爆炸
        self.update_embedding = True  # 训练的时候更新映射
        self.pretrain_embedding = False  # 词向量的初始化方式，随机初始化
        self.embedding_dim = 100  # 词向量的维数
        self.shuffle = True  # 打乱训练数据
        self.log = 'train_log.txt'  # 日志保存路径

    @property
    def get_log(self):
        return self.log

    @property
    def get_root_data_path(self):
        return self.root_data_path

    @property
    def get_batch_size(self):
        return self.batch_size

    @property
    def get_epoch(self):
        return self.epoch

    @property
    def get_hidden_dim(self):
        return self.hidden_dim

    @property
    def get_optimizer(self):
        return self.optimizer

    @property
    def get_lr(self):
        return self.lr

    @property
    def get_clip(self):
        return self.clip

    @property
    def get_update_embedding(self):
        return self.update_embedding

    @property
    def get_pretrain_embedding(self):
        return self.pretrain_embedding

    @property
    def get_embedding_dim(self):
        return self.embedding_dim

    @property
    def get_shuffle(self):
        return self.shuffle


class Train:

    def __init__(self, model_name):
        self.config = Config()
        self.dataparse = DataParse()
        self.model_name = model_name
        self.au_vocab, self.key_vocab, self.au_len, self.key = info('DataProcess/data_manage_key_mapping.pkl')
        if not self.config.get_pretrain_embedding:
            self.embeddings_au = self.dataparse.random_embedding(self.config.get_embedding_dim, self.au_len)  # author embedding init
            self.embeddings_key = self.dataparse.random_embedding(self.config.get_embedding_dim, self.key)  # keywords embedding init
        else:
            print('Error embedding loading')
        self.model_same_path = self.config.get_root_data_path.__add__('/').__add__(model_name).__add__('/')  # 模型路径

    def train_and_eva(self):  # 越界。
        model = Bi_LSTM_Task(param_config=self.config,
                             embeddings_au=self.embeddings_au,
                             embeddings_key=self.embeddings_key,
                             vocab_au=self.au_vocab,
                             vocab_key=self.key_vocab,
                             model_save_path=self.model_same_path,
                             au_class=self.au_len
                            )  # 详见bi_lstm_model.py里面的注释
        model.build_graph()  # 创建网络时，就已经计算了网络的损失
        print('net created......')
        out_put_log = open(self.config.get_log, mode='w', encoding='utf-8', newline='')
        model.train_and_eva(log_file=out_put_log)


if __name__ == '__main__':

    train_pro = Train(model_name='task')
    train_pro.train_and_eva()