import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from data_parse import Data_Inter


def check_multi_path(path):
    # 如果路径包含多个不存在的文件夹，对此路径继续宁操作会报路径不存在错误，本方法的作用
    # 检查路径是否合法，不合法，那么根据路径创建文件夹，使得路径合法。
    assert isinstance(path, str) and len(path) > 0
    if '\\' in path:
        path.replace('\\', '/')
    childs = path.split('/')
    root = childs[0]
    for index, cur_child in enumerate(childs):  # enumerate：给列表的每一个元素附加其所在的索引。
        if index > 0:
            root = os.path.join(root, cur_child)
        if not os.path.exists(root):
            os.mkdir(root)


class Bi_LSTM_Task:
    def __init__(self, param_config,  # config，配置一些常用的参数
                 embeddings_au,  # 总的嵌入
                 embeddings_key,  # 总的嵌入
                 vocab_au,  # 字典
                 vocab_key,  # 字典
                 model_save_path,
                 au_class
                 ):  # 同上
        self.batch_size = param_config.get_batch_size
        self.epoch_num = param_config.get_epoch
        self.hidden_dim = param_config.get_hidden_dim
        self.embeddings_au = embeddings_au
        self.embeddings_key = embeddings_key
        self.update_embedding = param_config.get_update_embedding
        # self.dropout_keep_prob = param_config.get_dropout
        self.optimizer = param_config.get_optimizer
        self.lr = param_config.get_lr
        self.clip_grad = param_config.get_clip
        self.vocab_au = vocab_au
        self.vocab_key = vocab_key
        self.au_class = au_class
        self.shuffle = param_config.get_shuffle
        self.model_path = model_save_path
        self.data_inter = Data_Inter(self.vocab_au, self.vocab_key)

    def build_graph(self):
        self.add_placeholders()  # 神经网络的占位符
        self.lookup_layer_op()  # 字典内所有分词的向量初始化
        self.biLSTM_layer_op()  # 神经网络的主体
        self.loss_op()  # 获取损失
        self.trainstep_op()  # 梯度求解

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[self.batch_size, 7], name="word_ids")
        self.word_ids_author = tf.placeholder(tf.int32, shape=[1, self.batch_size], name="word_ids_author")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.task_targets_Au = tf.placeholder(tf.int64, [self.batch_size], name='task_targets_Au')  # 16

    def lookup_layer_op(self):
        """
        将词的one-hot形式表示成词向量的形式，词向量这里采用随机初始化的形式，显然可以使用w2c预训练的词向量。
        """
        with tf.variable_scope("words"):
            _word_embeddings_au = tf.Variable(self.embeddings_au,
                                              dtype=tf.float32,
                                              trainable=self.update_embedding,
                                              name="_word_embeddings")
            self.word_embeddings_au = tf.nn.embedding_lookup(params=_word_embeddings_au,
                                                             ids=self.word_ids_author,
                                                             name="word_embeddings")
            #
            _word_embeddings_key = tf.Variable(self.embeddings_key,
                                               dtype=tf.float32,
                                               trainable=self.update_embedding,
                                               name="_word_embeddings")
            self.word_embeddings_key = tf.nn.embedding_lookup(params=_word_embeddings_key,
                                                              ids=self.word_ids,
                                                              name="word_embeddings")
            print('embedding of author and keys:', self.word_embeddings_au, self.word_embeddings_key)

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        print('shape of H', H.shape)
        H = tf.expand_dims(H, axis=-1)  # 32 * 1 * 200
        hiddenSize = 200
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([1, hiddenSize], stddev=0.1))
        tmp = W
        for j in range(H.shape[0] - 1):
            tmp = tf.concat((tmp, W), axis=0)
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        print('M', M.shape)
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        # newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))  # 32 * 200 * 1 * 200
        newM = tf.multiply(tf.reshape(M, [-1, hiddenSize]), tmp)  # 32 * 200 * 1 * 200
        print('newM:', newM.shape)
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, 200])
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.multiply(H, tf.reshape(self.alpha, [-1, 200, 1]))
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        sentenceRepren = tf.tanh(sequeezeR)
        return sentenceRepren

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim // 2, name='an_fw')
            cell_bw = LSTMCell(self.hidden_dim // 2, name='an_bw')
            (output_fw_seq, output_bw_seq), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=self.word_embeddings_au,  # author embedding
                        dtype=tf.float32)
            output_au = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            encoder_final_state_h_au = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
            # (32, 2, 200)
            print('au embedding:', output_au, encoder_final_state_h_au)
            cell_fw = LSTMCell(self.hidden_dim // 2, name='key_fw')
            cell_bw = LSTMCell(self.hidden_dim // 2, name='key_bw')
            (output_fw_seq_key, output_bw_seq_key), (encoder_fw_final_state_key, encoder_bw_final_state_key) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=self.word_embeddings_key,  # author embedding
                    dtype=tf.float32)
            output_key = tf.concat([output_fw_seq_key, output_bw_seq_key], axis=-1)
            encoder_final_state_h_key = tf.concat((encoder_fw_final_state_key.h, encoder_bw_final_state_key.h), 1)

            # 32, 7, 200
            # print('key embedding:', output_key, encoder_final_state_h_key, encoder_final_state_h_key_)
            print('key embedding:', output_key, encoder_final_state_h_key)
            # feature concat
            merged_feature = tf.concat((output_au[0, :, :], encoder_final_state_h_key), axis=1)
            print('merged_feature:', merged_feature)
            # att
            att_fea = self.attention(merged_feature)


        with tf.variable_scope("proj"):
            w_task_kw = tf.get_variable(name="w_task_kw",  # w_task_kw与encoder_final_state_h进行矩阵乘法，得到对类别的映射
                                        shape=[4 * self.hidden_dim, self.au_class],  # intent的个数
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        dtype=tf.float32)
            b_task_kw = tf.get_variable(name="b_task_kw",
                                        shape=[self.au_class],
                                        initializer=tf.zeros_initializer(),
                                        dtype=tf.float32)
            # task
            logits = tf.add(tf.matmul(merged_feature, w_task_kw), b_task_kw)  # 异构网络1
            print('logits:', logits)
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                     labels=tf.one_hot(self.task_targets_Au,
                                                                                       depth=self.au_class))
            self.pred_loss = tf.reduce_mean(self.pred_loss)
            self.regular_train_op = tf.train.AdamOptimizer(self.lr_pl).minimize(self.pred_loss)
            self.correct_label_pred = tf.equal(self.task_targets_Au, tf.argmax(self.pred, 1))
            self.label_acc = tf.reduce_mean(tf.cast(self.correct_label_pred, tf.float32))

    def loss_op(self):
            self.loss = self.pred_loss  # 任务识别的损失

    def trainstep_op(self):
        """
        训练节点.
        """
        with tf.variable_scope("train_step"):  # 优化函数的选择，没什么好注释的。本文使用的是 Adam
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局训批次的变量，不可训练。
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)  # 计算损失
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)  # 收集所有可训练的参数，进行训练。clip_grad限制梯度范围

    def pad_sequences(self, sequences, pad_mark=0, predict=False, vocab_flag=0):
        """
        批量的embedding，其中rowtext embedding的长度要与slots embedding的长度一致，不然使用crf时会出错。
        :param sequences: 批量的文本格式[[], [], ......, []]，其中子项[]里面是一个完整句子的embedding（索引。）
        :param pad_mark:  长度不够时，使用何种方式进行padding
        :param vocab_flag:  长度不够时，使用何种方式进行padding
        :param predict:  是否是测试
        :return:
        """
        # print('sequences:', sequences)
        max_len = max(map(lambda x: len(x), sequences))  # 当前输入批量样本的最大长度，以此为基准扩充样本长度。padding。什么是padding，请问度娘
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            # print('传进来的数据:', seq)
            # if predict:
            #     seq = list(map(lambda x: self.vocab.get(x, 0), seq))  # 获取索引，依据字典
            if vocab_flag == 0:  # au
                seq_ = seq[:len(seq)] + [self.vocab_au['A_NA']] * max(max_len - len(seq), 0)  # 截断和填充
            else:  # au
                seq_ = seq[:len(seq)] + [self.vocab_key['PAD']] * max(max_len - len(seq), 0)  # 截断和填充
            seq_list.append(seq_)  # 收集
            seq_len_list.append(min(len(seq), max_len))  # 保留当前样本的实际长度
        return seq_list, seq_len_list

    def train_and_eva(self, log_file=None):
        """
            数据由一个外部迭代器提供。
        """
        test_save = open('test.txt', mode='w', encoding='utf-8')
        log_file = open('train_log.txt', mode='w', encoding='utf-8')
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt_file = tf.train.latest_checkpoint('model_save/'.__add__('/task/'))
            if ckpt_file is not None:
                saver.restore(sess, ckpt_file)
            for epoch_index in range(0, self.epoch_num, 1):
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sentence, task_ = self.data_inter.next()  # 迭代器，每次取出一个batch块.
                feed_dict, _ = self.get_feed_dict(sentence, task_, self.lr)
                _, loss_train, acc_, step_num_ = sess.run([self.train_op,
                                                           self.loss,
                                                           self.label_acc,
                                                           self.global_step],
                                                          feed_dict=feed_dict)
                if epoch_index % 200 == 0 and epoch_index != 0:
                    if log_file is not None:
                        log_file.write('time:'.__add__(start_time).
                                       __add__('\tstep:').
                                       __add__(str(epoch_index)).
                                       __add__('\tloss:').__add__(str(loss_train)).
                                       __add__('\tacc:').__add__(str(acc_)).
                                       __add__('\n'))
                    # print('time {}, step {}, loss: {:.4}, acc_: {:.4}'.
                    #       format(start_time, epoch_index, loss_train, acc_))
                    check_multi_path(self.model_path)
                    saver.save(sess, self.model_path, global_step=epoch_index)
                if epoch_index % 50 == 0:
                    # Test Stage
                    sentence_test, task_test = self.data_inter.next_test()  # 迭代器，每次取出一个batch块.
                    feed_dict_test, _ = self.get_feed_dict(sentence_test, task_test)
                    loss_test_test, acc_test = sess.run([
                                                     self.loss,
                                                     self.label_acc],
                                                    feed_dict=feed_dict_test)
                    test_save.write('time:'.__add__(start_time).
                                    __add__('\tepoch: ').  # 写入本地日志
                                    __add__('\ttest loss:').
                                    __add__(str(loss_test_test)).
                                    __add__('\ttest acc:').
                                    __add__(str(acc_test)).
                                    __add__('\n'))
                    print('time {}, step {}, test_loss: {:.4}, test_acc: {:.4}'.
                          format(start_time, epoch_index, loss_test_test, acc_test))
                # ######################################################################################

            if log_file is not None:
                log_file.close()  # 关闭输出流

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None, predicted=False):
        """

        :param seqs:  训练的batch块
        :param labels:  实体标签
        :param lr:  学利率
        :param dropout:  活跃的节点数，全连接层
        :return: feed_dict  训练数据
        :return: predicted  测试标志
        """
        word_ids, seq_len_list = self.pad_sequences(seqs, pad_mark=0, predict=predicted, vocab_flag=1)
        word_ids_au, seq_len_list_au = self.pad_sequences(labels, pad_mark=0, predict=predicted, vocab_flag=0)
        word_ids_au = np.array(word_ids_au)
        feed_dict = {
            self.word_ids: word_ids,  # embedding到同一长度
            self.word_ids_author: np.expand_dims(word_ids_au[:, 0], axis=0),  # embedding到同一长度
            self.sequence_lengths: seq_len_list,  # 实际长度。
                     }
        if labels is not None:
            feed_dict[self.task_targets_Au] = word_ids_au[:, 1]
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        return feed_dict, seq_len_list

    def predict(self, sess, seqs, predicted=False):  # 预测
        """

        :param sess:
        :param seqs:
        :param predicted:
        :return: label_list
                 seq_len_list
        """
    pass



