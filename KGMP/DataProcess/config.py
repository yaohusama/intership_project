class ConfigParam:
    def __init__(self):
        self.seta = 1.1
        self.beta = 1.5
        self.epoch = 200
        self.get_clip = 1000.
        self.batch_size = 32
        self.repost = 21
        self.learning_rate = .001
        self.keep_dropout = .5
        self.optimizer = 'Adam'
        self.shuffle = True
        self.sequence_length = 80
        self.embedding_size = 128
        self.get_embedding_dim = 128
        self.update_embedding = True  # 训练的时候更新映射
        self.model_saved_path = '../model/task/'
        self.logging_file_saved_path = '../log_file/news_'

