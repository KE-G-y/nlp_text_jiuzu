import torch
import os
from transformers.models import BertModel,BertTokenizer,BertConfig
#配置文件
class Config:
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型训练+预测的时候
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu
        self.num_epochs = 2  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = os.path.join(current_dir, "bert-base-chinese")  # 预训练BERT模型的路径（绝对路径）
        self.bert_model = BertModel.from_pretrained(self.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path) # BERT模型的配置
        self.hidden_size = 768 # BERT模型的隐藏层大小

conf = Config()


if __name__ == '__main__':
    print(conf)


