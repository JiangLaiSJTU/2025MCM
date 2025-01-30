import pandas as pd

def load_and_preprocess_data(file_path):
    # 读取数据
    data = pd.read_csv(file_path)
    print("原始数据集的前几行：")
    print(data.head())

    # 清洗数据，排除 'Mixed team' 的行
    data = data[data['NOC'] != 'Mixed team']
    print("\n清洗后的数据集的前几行：")
    print(data.head())

    # 检查列名，去除列名中的空格
    data.columns = data.columns.str.strip()
    print("\n去除空格后的列名：", data.columns.tolist())

    # 按年份排序数据
    data = data.sort_values(by=['NOC', 'Year'])
    print("\n按年份排序后的数据集的前几行：")
    print(data.head())

    return data 