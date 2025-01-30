import pandas as pd

# 读取原始数据（根据实际文件路径修改）
medal_counts = pd.read_csv('./data/summerOly_medal_counts.csv')  # 包含NOC列（国家全称）
athletes = pd.read_csv('./data/summerOly_athletes.csv')          # 包含Team列（国家全称）

# --------------------------
# 步骤1：提取2024年未获奖国家
# --------------------------

# 筛选2024年的奖牌数据
medal_2024 = medal_counts[medal_counts['Year'] == 2024]

# 获取所有获奖国家的全称（NOC列）
medaled_countries_2024 = medal_2024['NOC'].unique().tolist()

# 筛选2024年的运动员数据
athletes_2024 = athletes[athletes['Year'] == 2024]

# 提取所有参赛国家的全称（Team列）
all_participant_countries = athletes_2024['Team'].unique().tolist()

# 找出未获奖国家（在参赛国家中但不在获奖国家中）
unmedaled_countries = list(set(all_participant_countries) - set(medaled_countries_2024))

# 生成未获奖国家列表
unmedaled_countries_df = pd.DataFrame({'Country': unmedaled_countries})

# 保存结果
unmedaled_countries_df.to_csv('2024_unmedaled_countries.csv', index=False)

# --------------------------
# 步骤2：提取未获奖国家的运动员信息
# --------------------------

# 筛选未获奖国家的运动员数据
unmedaled_athletes = athletes_2024[
    athletes_2024['Team'].isin(unmedaled_countries) &
    (athletes_2024['Medal'] == 'No medal')
]

# 选择需要的列
columns_to_keep = ['Name', 'Sex', 'Team', 'Sport', 'Event', 'Medal']
unmedaled_athletes = unmedaled_athletes[columns_to_keep]

# 保存结果
unmedaled_athletes.to_csv('2024_unmedaled_athletes.csv', index=False)

print('生成文件完成：')
print('- 未获奖国家列表：2024_unmedaled_countries.csv')
print('- 未获奖运动员信息：2024_unmedaled_athletes.csv')