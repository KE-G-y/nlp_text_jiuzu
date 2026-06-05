from configuration.config import *
from datasets import load_dataset
from datasets import ClassLabel
from transformers import AutoTokenizer


def preprocess():
    # 1.读取数据   从文件加载数据集
    dataset_dict = load_dataset("csv",  # csv 默认逗号分割
                                data_files={"train": str(RAW_DATA_DIR / RAW_TRAIN_DATA),
                                            'valid': str(RAW_DATA_DIR / RAW_VALID_DATA),
                                            "test": str(RAW_DATA_DIR / RAW_TEST_DATA)
                                            },
                                delimiter="\t",  # 指定分隔符
                                column_names=["label", "text_a"],  # 指定列名
                                skiprows=1,  # 跳过表头行
                                )

    # print(dataset_dict)
    #
    # # 2. 过滤数据       当两个字段都存在且不为 None 时，才保留该样本
    dataset_dict = dataset_dict.filter(lambda x: x['text_a'] is not None and x['label'] is not None)
    #
    # # 3. 处理类别   label编码处理
    # 3.1 获取所有label字段返回一个列表  set转集合去重  sorted排序
    all_labels = sorted(set(dataset_dict['train']['label']))

    # 3.2 创建标签到索引的映射
    label2id = {label: idx for idx, label in enumerate(all_labels)}

    # 3.3 将字符串标签转换为整数索引
    def convert_label(example):
        example['label'] = label2id[example['label']]
        return example

    dataset_dict = dataset_dict.map(convert_label)

    # 3.4 列转换编码  label列类型转换,指定转换为all_labels映射为数字索引
    dataset_dict = dataset_dict.cast_column('label', ClassLabel(names=all_labels))

    # 查看前三条数据
    # print(dataset_dict['train'][:3])

    # 2.3 将映射关系保存 id -> label
    with open(MODEL_DIR / LABEL_FILE, 'w', encoding='utf-8') as f:
        for label in all_labels:
            f.write(f'{label}\n')

    # 3.加载分词器   # AutoTokenizer 会根据模型名称自动选择合适的分词器类型  将中文文本转换为 BERT 模型能理解的 token ID
    tokenizers = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 4.处理标题文本
    def batch_encode(examples): # 接收一个批次的数据样本
        # 将本字段 text_a 转换为 token IDs  truncation 如果文本超过最大长度（默认512），自动截断
        inputs = tokenizers(examples['text_a'], truncation=True)
        inputs['labels'] = examples['label']    # 将 label 字段添加到输出中（用于训练）
        return inputs   # 回包含 input_ids、attention_mask、token_type_ids 的字典

    # map:数据集中的每一条样本应用指定的函数
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['label', 'text_a'])
    print(dataset_dict['train'][:3])

    # 5. 保存数据集
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == '__main__':
    preprocess()
