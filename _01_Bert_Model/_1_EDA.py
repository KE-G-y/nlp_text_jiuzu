"""
数据EDA：查看数据
    1.导入包
    2.获取文件路径
    3.读取数据并查看基本信息
"""
# 1.导入包
from _0_config import Config
import pandas as pd
from collections import Counter
config = Config()
"""Counter 类，本身是一个字典子类，它返回的是一种以可哈希对象为键、以出现次数（整数）为值的字典格式数据。"""

# 2.处理数据集格式
import pandas as pd


def build_label_text(origin_path, encoding='utf-8'):
    """
    读取 label+空格+text 格式的文件，text内部可能有大量空格
    参数：
        file_path: 文件路径
        encoding: 文件编码（如 'utf-8' 或 'gbk'）
    返回：
        DataFrame，包含两列：'label' 和 'text'
    """
    labels = []  # 用来存放所有 label 的列表
    texts = []  # 用来存放所有 text 的列表

    with open(origin_path, 'r', encoding="utf-8") as f:
        # f = f.iloc[1:]  # 方法1：切片保留第2行及以后
        # line_num 是行号，从1开始计数，方便报错时定位
        for line_num, line in enumerate(f, 1):  # 会给每一行自动编号，从 1 开始
            line = line.strip()  # 去掉行首行尾的空白字符（如换行符、空格）
            if not line:  # 如果去掉空白后是空行，则跳过
                continue

            # ***** 最关键的一行 *****
            # split(maxsplit=1) 表示：只按空白字符（空格、制表符等）分割一次
            # 结果 parts 是一个列表，长度为2：[label, 剩下的全部内容]
            parts = line.split(maxsplit=1)

            if len(parts) == 2:  # 正常情况：有 label 和 text
                labels.append(parts[0])  # 第一部分是 label
                parts[1] = parts[1].replace(' ', '')
                texts.append(parts[1])  # 第二部分是 text（可能含内部空格）

    data = pd.DataFrame({'label': labels, 'text': texts})
    data = data.iloc[1:]
    # 用两个列表构建 DataFrame
    return data

"""训练数据集处理"""
train_data = build_label_text(config.train_path)
train_data.to_csv(f"{config.train_data}", index=False, encoding='utf-8')  # 保存数据
print(f'前5行数据：\n {train_data.head(5)}')
print(f'训练集数据量： {len(train_data)}')

"""测试数据集处理"""
test_data = build_label_text(config.test_path)
test_data.to_csv(f"{config.test_data}", index=False, encoding='utf-8')  # 保存数据
print(f'前5行数据：\n {test_data.head(5)}')
print(f'训练集数据量： {len(test_data)}')


"""测试数据集处理"""
labels = []  # 用来存放所有 label 的列表
texts = []  # 用来存放所有 text 的列表

with open(config.dev_path, 'r', encoding="utf-8") as f:
    # f = f.iloc[1:]  # 方法1：切片保留第2行及以后
    # line_num 是行号，从1开始计数，方便报错时定位
    for line_num, line in enumerate(f, 1):  # 会给每一行自动编号，从 1 开始
        line = line.strip()  # 去掉行首行尾的空白字符（如换行符、空格）
        if not line:  # 如果去掉空白后是空行，则跳过
            continue

        # ***** 最关键的一行 *****
        # split(maxsplit=1) 表示：只按空白字符（空格、制表符等）分割一次
        # 结果 parts 是一个列表，长度为2：[label, 剩下的全部内容]
        parts = line.rsplit(maxsplit=1)

        if len(parts) == 2:  # 正常情况：有 label 和 text
            labels.append(parts[1])  # 第二部分是 label
            parts[0] = parts[0].replace(' ', '')
            texts.append(parts[0])  # 第二部分是 text（可能含内部空格）

dev_data = pd.DataFrame({'label': labels, 'text': texts})
dev_data.to_csv(f"{config.dev_data}", index=False, encoding='utf-8')  # 保存数据
print(f'前5行数据：\n {dev_data.head(5)}')
print(f'验证集数据量： {len(dev_data)}')


# 3.样本均衡性---分类分布
label_count = Counter(train_data['label'])  # 返回一个字典
print(label_count.items())
for label, count in label_count.items():
    print(f'{label}: {count}')
    print(f'{label}:{count/len(train_data) * 100 :.2f}%')



# 4.数据分布--文本长度== 找到最优文本长度
# 统计每行文本的长度
# train_data['text_length'] = (train_data['text'].str.len())
# print(f"前10行数据:{train_data.head(10)}")
#
# # 统计文本长度的分类  -- 均值、标准差、最大值、最小值
# print(train_data["text_length"].mean())
# print(train_data["text_length"].std())
# print(train_data["text_length"].max())
# print(train_data["text_length"].min())