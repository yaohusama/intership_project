import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

p_global = 'train_log.txt'
p_global_test = 'test.txt'


def read_result():

    # cur trainer:trainer_global	batch_loss:2.310460329055786	batch_acc:0.0
    res = {
        'p_global':
            {
                'step': [],
                'loss': [],
                'acc': [],
            },
        'p_global_test': {
            'step': [],
            'loss': [],
            'acc': [],
    }}
    for cur_path, name_ in zip([p_global], ['p_global']):
        datas = open(cur_path, mode='r', encoding='utf-8').readlines()
        for index_cur_data, cur_data in enumerate(datas):
            if index_cur_data > 200:
                break
            else:
                batch_loss, acc_fw, acc_hj, acc_kw = list(map(lambda x: float(x.split(':')[-1]), cur_data.strip().split('\t')[3:]))[:]
                res[name_]['step'].append(index_cur_data)
                res[name_]['loss'].append(batch_loss)
                res[name_]['acc'].append(acc_fw)
    #
    return res


def tensorboard_data_generate(computed_data):
    batch_loss_source_global = tf.placeholder(tf.float32, None, name="Batch_Loss_Train")
    batch_acc_source_global = tf.placeholder(tf.float32, None, name="Batch_Acc_Train")

    batch_loss_source_global_test = tf.placeholder(tf.float32, None, name="Batch_Loss_Test")
    batch_acc_source_global_test = tf.placeholder(tf.float32, None, name="Batch_Acc_FW_Test")

    init = tf.global_variables_initializer()

    tf.summary.scalar("Batch_Loss_Train", batch_loss_source_global)
    tf.summary.scalar("Batch_Acc_FW_Train", batch_acc_source_global)

    # #########################################################################################
    #
    # tf.summary.scalar("Batch_Loss_Train", batch_loss_source_global_test)
    # tf.summary.scalar("Batch_Loss_Test", batch_acc_source_global_test)
    # tf.summary.scalar("Batch_Acc_HJ_Test", batch_loss_domain_global_test)
    # tf.summary.scalar("Batch_Acc_KW_Test", batch_acc_domain_global_test)


    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("board_res/", graph_def=sess.graph_def)
        sess.run(init)
        step_global = computed_data["p_global"]['step']
        loss_source_global = computed_data["p_global"]['loss']
        loss_domain_global = computed_data["p_global"]['acc']
        # ################################################################################################################
        #
        for m, n, x, y, p in zip(loss_source_global, loss_domain_global, step_global):
            summary_str = sess.run(summary_op, feed_dict={batch_loss_source_global: m,
                                                          batch_acc_source_global: n,
                                                          })
            summary_writer.add_summary(summary_str, p)


if __name__ == '__main__':
    # 生成 tersorboard 数据
    tensorboard_data_generate(read_result())
    # tensorboard_data_generate(tcn_loss, tcn_acc, len(pro_acc), "pro_loss", "pro_acc")
    # 生成 plt数据
    # all_res = read_result()
    # plt_type_data(all_res)
