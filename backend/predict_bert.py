# src/runner/predict.py
# 模型预测脚本 —— 加载训练好的模型，对新文本进行类别预测
'''
todo:预测脚本
'''
import json
import torch
from transformers import AutoTokenizer
from pathlib import Path

from config import conf
from classifier import BertTitleClassifier

BASE_DIR = Path(__file__).parent


def load_model():
    """加载训练好的模型、分词器和类别映射（train.py 保存的三样东西）"""
    # 1. 读取模型元信息（类别数 + 类别名列表）
    with open(BASE_DIR / 'model_info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    num_classes = info['num_classes']
    class_names = info['class_names']  # ['手机', '衣服', '食品', ...]

    # 2. 加载分词器
    tokenizer = conf.tokenizer

    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertTitleClassifier(num_classes=num_classes).to(device)
    MODEL_PATH = BASE_DIR / 'models' / 'model_bert.pt'
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model.eval()

    return model, tokenizer, class_names, device


# 全局加载，只执行一次
model, tokenizer, class_names, device = load_model()


def predict(text):
    """对单条商品标题文本进行类别预测

    参数:
        text: str, 商品标题文本，如 "iPhone 15 Pro Max 256GB"

    返回:
        dict: {'text': 原文本, 'pred_class': 预测类别名}
    """
    # 1. 分词：text → input_ids + attention_mask
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 2. 前向推理（关闭梯度计算，节省显存）
    with torch.no_grad():
        logits = model(input_ids, attention_mask)          # (1, num_classes)
        pred_id = torch.argmax(logits, dim=1).item()       # 概率最大的类别索引
        pred_class = class_names[pred_id]                  # 索引 → 类别名

    return {'text': text, 'pred_class': pred_class}


if __name__ == '__main__':
    # 测试：预测单个商品标题
    result = predict('黑兰州')
    print(f"标题: {result['text']}")
    print(f"预测类别: {result['pred_class']}")
