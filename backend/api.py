# 模型预测
from flask import Flask, jsonify, request

from predict_bert import predict as predict_bert
from predict_ft import predict as predict_fasttext

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json() or {}
    model_name = data.get('model') or 'fasttext_char'

    try:
        if model_name in {'fasttext_char', 'fasttext_word'}:
            result = predict_fasttext(data)
        elif model_name == 'bert':
            bert_result = predict_bert(data.get('text', ''))
            result = {**data, **bert_result, 'model': model_name}
        else:
            return jsonify({'error': f'不支持的模型：{model_name}'}), 400
    except Exception as error:
        return jsonify({'error': str(error), 'model': model_name}), 500

    print('-------------预测结果------------')
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
