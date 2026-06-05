# 模型预测
import warnings
from pathlib import Path

import fasttext

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
MODEL_PATHS = {
    'fasttext_char': BASE_DIR / 'models' / 'ft_char.bin',
    'fasttext_word': BASE_DIR / 'models' / 'ft_word.bin',
}
MODELS = {name: fasttext.load_model(str(path)) for name, path in MODEL_PATHS.items()}


def predict(data):
    model_name = data.get('model') or 'fasttext_char'
    if model_name not in MODELS:
        raise ValueError(f'不支持的 FastText 模型：{model_name}')

    text = data['text']
    words = ' '.join(list(text)) if model_name == 'fasttext_char' else text
    res = MODELS[model_name].predict(words)

    if res and len(res) > 0 and res[0] and len(res[0]) > 0:
        result = res[0][0][9:] if len(res[0][0]) > 9 else res[0][0]
    else:
        result = ''

    data['model'] = model_name
    data['pred_class'] = result
    return data

if __name__ == '__main__':
    data = {"text": "好奇心钻装纸尿裤L40片9-14kg"}
    print(predict(data))
