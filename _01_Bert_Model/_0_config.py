import torch
import os
import datetime



current_date=datetime.datetime.now().date().strftime("%Y%m%d")

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        # self.model_name = "bert" # 模型名称
        self.data_root = "../data"  #数据集的根路径
        self.train_path = self.data_root + "\\train.txt"  # 训练集
        self.dev_path = self.data_root + "\\valid.txt"  # 少量验证集，快速验证
        self.test_path = self.data_root + "\\test.txt"  # 测试集
        # 处理后的文档
        self.train_data = self.data_root + "\\train_data.txt" # 训练集
        self.dev_data = self.data_root + "\\dev_data.txt"  # 少量验证集，快速验证
        self.test_data = self.data_root + "\\test_data.txt"  # 测试集

        # self.class_path=self.data_path + "\\class.txt" #类别文件
        # self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        self.model_save_path = "Fast_model_save/train_model.pt"  #模型训练结果保存路径

        # 模型训练+预测的时候
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu

        # self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率


if __name__ == '__main__':
    conf = Config()
