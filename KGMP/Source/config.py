class Config:
    def __init__(self):
        self.epoch = 200  # 训练周期，就是数据集轮几遍
        self.get_clip = 10.  # 防止梯度爆炸，限制总梯度不大于10
        self.batch_size = 32  # 一次性宋玉到神经网络的数据大小
        self.learning_rate = .0001  # 学习率
        self.keep_dropout = .5  # 随即激活概率
        self.optimizer = 'Adam'  # 优化函数
        self.shuffle = True  # 是否打乱数据，数据越无序，训练的会越充分
        self.sequence_length = 64  # 文本的长度，文本太长需要截断，太短不需要，这个参数本文没用到
        self.embedding_size = 32  # 每个分词嵌入的大小，就是每个分词用一个多少维的向量表示
        self.model_saved_path = '../model/task/'  # 超参数的保存路径
        self.logging_file_saved_path = '../log_file/'  # 训练过程中日志文件的保存路径

