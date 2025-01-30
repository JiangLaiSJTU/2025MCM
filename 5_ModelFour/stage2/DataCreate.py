# # import csv
# #
# #
# # def filter_rows_by_year(input_file, output_file, target_year):
# #     """
# #     从CSV文件中筛选出Year列值为目标年份的行，并保存到新的文件中。
# #
# #     :param input_file: 输入的CSV文件路径
# #     :param output_file: 输出的CSV文件路径
# #     :param target_year: 目标年份
# #     """
# #     try:
# #         # 打开输入文件进行读取
# #         with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
# #             reader = csv.DictReader(infile)
# #             # 获取表头
# #             headers = reader.fieldnames
# #
# #             # 打开输出文件进行写入
# #             with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
# #                 writer = csv.DictWriter(outfile, fieldnames=headers)
# #                 writer.writeheader()  # 写入表头
# #
# #                 # 遍历输入文件中的每一行
# #                 for row in reader:
# #                     # 检查Year列是否为目标年份
# #                     if row['Year'] == str(target_year):
# #                         writer.writerow(row)  # 写入符合条件的行
# #
# #         print(f"筛选完成，结果已保存到 {output_file}")
# #     except Exception as e:
# #         print(f"发生错误：{e}")
# #
# #
# # # 设置输入和输出文件路径
# # input_file_path = 'summerOly_athletes.csv'  # 原始文件名
# # output_file_path = 'new.csv'  # 新文件名
# # target_year = 2024  # 目标年份
# #
# # # 调用函数进行筛选
# # filter_rows_by_year(input_file_path, output_file_path, target_year)
# import csv
# from collections import Counter
#
#
# def filter_rows_by_year_and_noc(input_file, output_file, target_year, min_noc_count, min_noc_medal_ratio):
#     """
#     从CSV文件中筛选出Year列值为目标年份的行，并删除出现少于指定次数的NOC相关的行，
#     同时删除NOC对应的Medal项中No medal占比超过指定比例的行，
#     最后将结果保存到新的文件中。
#
#     :param input_file: 输入的CSV文件路径
#     :param output_file: 输出的CSV文件路径
#     :param target_year: 目标年份
#     :param min_noc_count: 最小NOC出现次数
#     :param min_noc_medal_ratio: NOC对应的Medal项中No medal的最大占比比例
#     """
#     try:
#         # 第一步：读取数据并筛选出Year为目标年份的行
#         with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
#             reader = csv.DictReader(infile)
#             headers = reader.fieldnames
#
#             # 筛选出Year为目标年份的行
#             filtered_rows = [row for row in reader if row['Year'] == str(target_year)]
#
#         # 第二步：统计每个NOC出现的次数
#         noc_counter = Counter(row['NOC'] for row in filtered_rows)
#
#         # 第三步：统计每个NOC对应的No medal占比
#         noc_medal_ratio = {}
#         for noc in noc_counter:
#             noc_rows = [row for row in filtered_rows if row['NOC'] == noc]
#             total_medals = len(noc_rows)
#             no_medal_count = sum(1 for row in noc_rows if row['Medal'] == 'No medal')
#             noc_medal_ratio[noc] = no_medal_count / total_medals if total_medals > 0 else 0
#
#         # 第四步：筛选出符合条件的行
#         final_rows = [
#             row for row in filtered_rows
#             if noc_counter[row['NOC']] >= min_noc_count and noc_medal_ratio[row['NOC']] <= min_noc_medal_ratio
#         ]
#
#         # 第五步：将结果写入新的文件
#         with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
#             writer = csv.DictWriter(outfile, fieldnames=headers)
#             writer.writeheader()
#             writer.writerows(final_rows)
#
#         print(f"筛选完成，结果已保存到 {output_file}")
#     except Exception as e:
#         print(f"发生错误：{e}")
#
#
# # 设置输入和输出文件路径
# input_file_path = 'new.csv'  # 原始文件名
# output_file_path = 'new1.csv'  # 新文件名
# target_year = 2024  # 目标年份
# min_noc_count = 40  # 最小NOC出现次数
# min_noc_medal_ratio = 0.90  # NOC对应的Medal项中No medal的最大占比比例
#
# # 调用函数进行筛选
# filter_rows_by_year_and_noc(input_file_path, output_file_path, target_year, min_noc_count, min_noc_medal_ratio)

# import pandas as pd
#
# # 读取原始CSV文件
# input_file = 'summerOly_athletes.csv'  # 原始数据文件路径
# output_file = 'filtered_data.csv'  # 筛选后的数据文件路径
#
# # 使用pandas读取CSV文件
# data = pd.read_csv(input_file)
#
# # 筛选条件：Year为2024或2020，Sport为"Judo"，Team为"Japan"
# filtered_data = data[(data['Year'].isin([2024, 2020])) &
#                      (data['Sport'] == 'Judo') &
#                      (data['Team'] == 'Japan')]
#
# # 将筛选后的数据保存到新的CSV文件中
# filtered_data.to_csv(output_file, index=False)
#
# print(f"筛选后的数据已保存到 {output_file}")

import pandas as pd

# 读取原始CSV文件
input_file = 'summerOly_athletes.csv'  # 原始数据文件路径
output_file = 'filtered_names_3years.csv'  # 输出文件路径

# 使用pandas读取CSV文件
data = pd.read_csv(input_file)

# 筛选条件：Sport为"Judo"，Team为"Japan"
filtered_data = data[(data['Sport'] == 'Judo') & (data['Team'] == 'Japan')]

# 分别提取2016、2020和2024年的数据，并去重
years = [2016, 2020, 2024]
results = {year: [] for year in years}

for year in years:
    year_data = filtered_data[filtered_data['Year'] == year]
    unique_names = year_data['Name'].drop_duplicates().tolist()
    results[year] = unique_names

# 找到所有年份中名字最多的数量，用于构造DataFrame
max_names = max(len(names) for names in results.values())

# 构造一个空的DataFrame
df = pd.DataFrame(index=range(1, max_names + 1), columns=years)

# 填充DataFrame
for year, names in results.items():
    df[year] = names + [None] * (max_names - len(names))

# 保存到CSV文件
df.to_csv(output_file, index_label='Year')

print(f"筛选后的数据已保存到 {output_file}")