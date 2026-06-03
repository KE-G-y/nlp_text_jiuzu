# src/model/classifier.py
# 模型定义 —— 基于 BERT 的商品标题分类器

import torch
from torch import nn
from transformers import AutoModel

from config import conf



class BertTitleClassifier(nn.Module):
    """BERT 商品标题分类器

    架构：预训练 BERT 编码器 + 线性分类头
    输入：商品标题的 input_ids 和 attention_mask
    输出：每个类别的 logits（未经 softmax）

    使用方式：
        model = BertTitleClassifier(num_classes=10, freeze_bert=True)
        logits = model(input_ids, attention_mask)
        pred = logits.argmax(dim=-1)  # 取最大 logit 对应的类别
    """

    def __init__(self, num_classes: int, freeze_bert: bool = True):
        """
        参数:
            num_classes: 分类类别数，由训练数据中的标签种类决定
            freeze_bert: 是否冻结 BERT 参数
                - True  = 只训练分类头，适合快速实验或数据量较小时
                - False = 全量微调，训练更慢但通常效果更好
        """
        super().__init__()

        # 加载本地预训练的 BERT 模型（无需联网下载）
        # AutoModel 返回不带分类头的裸 BERT，输出 hidden_states (batch, seq_len, 768)
        self.bert = AutoModel.from_pretrained(
            str(conf.bert_path)
        )

        # 分类头：将 [CLS] 位置的 768 维向量映射到 num_classes 个类别
        # nn.Linear 包含权重矩阵 (num_classes, 768) 和偏置 (num_classes,)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 冻结 BERT 参数：将 requires_grad 设为 False
        # 这样优化器不会更新这些参数，只更新分类头
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播 —— 完成一次从 token id 到类别 logits 的完整计算

        参数:
            input_ids:      (batch_size, seq_len) 分词后的 token id 序列
            attention_mask: (batch_size, seq_len)  1=有效token / 0=[PAD] 填充位，BERT 会忽略 mask=0 的位置

        返回:
            logits: (batch_size, num_classes) 未归一化的分类得分，配合 CrossEntropyLoss 使用

        计算流程:
            1. BERT 编码 → (batch, seq_len, 768) 的 hidden_states
            2. 取 [CLS] token（第 0 个位置）的输出 → (batch, 768)
            3. 线性变换 → (batch, num_classes)
        """
        # BERT 前向传播
        # last_hidden_state 形状: (batch_size, seq_len, hidden_size=768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 取 [CLS] token 的 hidden state
        # BERT 在每个序列开头插入 [CLS]，其输出向量汇总了整个句子的语义信息
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # 线性映射到类别空间
        logits = self.classifier(cls_output)  # (batch_size, num_classes)

        return logits