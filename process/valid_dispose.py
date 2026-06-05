import pandas as pd
from pathlib import Path

# 定义路径
RAW_DATA_DIR = Path("E:/AI_work_space/Project_AI/product-classify/data/raw")
valid_file = RAW_DATA_DIR / "valid.txt"

# 读取文件（没有表头）
df = pd.read_csv(valid_file, sep='\t', header=None, names=['text_a', 'label'])

# 交换列顺序
df = df[['label', 'text_a']]

# 添加表头并保存
df.to_csv(valid_file, sep='\t', index=False, encoding='utf-8')

print(f"已处理 {len(df)} 行数据")
print("前3行示例：")
print(df.head(3))
