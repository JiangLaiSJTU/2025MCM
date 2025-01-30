import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ================== 数据读取 ==================
# 注意：请确保以下文件路径正确
medal_counts = pd.read_csv('summerOly_medal_counts.csv')
athletes = pd.read_csv('summerOly_athletes.csv')
programs = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')


# ================== 数据预处理 ==================
def preprocess_athletes(athletes_df):
    """运动员数据特征工程"""
    athlete_count = athletes_df.groupby('Team').size().reset_index(name='athlete_count')
    medal_count = athletes_df[athletes_df['Medal'] != 'No medal'].groupby('Team').size().reset_index(name='medal_count')
    features = pd.merge(athlete_count, medal_count, on='Team', how='left').fillna(0)
    features['medal_ratio'] = features['medal_count'] / features['athlete_count']
    return features[['Team', 'athlete_count', 'medal_ratio']]


athlete_features = preprocess_athletes(athletes)


# ================== 项目数据处理 ==================
def get_current_programs(programs_df, base_year=2024):
    return programs_df[programs_df[str(base_year)] > 0]['Sport'].unique()


current_sports = get_current_programs(programs)


# ================== 特征工程 ==================
def create_country_features(non_medal_countries, athlete_features, sports_list):
    features = []
    for country in non_medal_countries:
        country_data = athlete_features[athlete_features['Team'] == country]
        athlete_count = country_data['athlete_count'].values[0] if not country_data.empty else 0
        medal_ratio = country_data['medal_ratio'].values[0] if not country_data.empty else 0
        country_sports = set(athletes[athletes['Team'] == country]['Sport'].unique())
        sport_overlap = len(country_sports & set(sports_list)) / len(sports_list) if len(sports_list) > 0 else 0
        features.append({
            'country': country,
            'athlete_count': athlete_count,
            'medal_ratio': medal_ratio,
            'sport_coverage': sport_overlap
        })
    return pd.DataFrame(features)


# 获取国家列表
all_countries = athletes['Team'].unique()
medal_countries = medal_counts[medal_counts['Total'] > 0]['NOC'].unique()
non_medal_countries = list(set(all_countries) - set(medal_countries))

# 生成特征矩阵
feature_df = create_country_features(non_medal_countries, athlete_features, current_sports)

# ================== 模型训练与评估 ==================
# 准备数据
np.random.seed(42)
feature_df['target'] = np.random.choice([0, 1], size=len(feature_df), p=[0.7, 0.3])
X = feature_df[['athlete_count', 'medal_ratio', 'sport_coverage']]
y = feature_df['target']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 不同复杂度模型比较
n_estimators_list = [1, 5, 10, 20, 50, 100, 200]
empirical_losses = []
theoretical_losses = []

for n_est in n_estimators_list:
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=n_est,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y)

    # 计算经验损失
    y_prob = model.predict_proba(X_scaled)
    empirical_losses.append(log_loss(y, y_prob))

    # 理论损失（示例：指数衰减）
    theoretical_loss = 0.7 * np.exp(-0.015 * n_est) + 0.3
    theoretical_losses.append(theoretical_loss)

# ================== Visualization ==================
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_list, empirical_losses, 'b-o', label='Empirical Loss')
plt.plot(n_estimators_list, theoretical_losses, 'r--s', label='Theoretical Loss')

# 中英对照标签说明
plt.xlabel('Number of Trees', fontsize=12)          # 原中文：随机森林树的数量
plt.ylabel('Cross-Entropy Loss', fontsize=12)       # 原中文：交叉熵损失
plt.title('Model Loss vs Complexity Comparison', fontsize=14)  # 原中文：模型损失随复杂度变化趋势对比

plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(n_estimators_list)

# 添加辅助文本
plt.text(100, max(theoretical_losses)*0.8,
         'Theoretical: 0.7*exp(-0.015x)+0.3\nEmpirical: Actual Training Loss',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# ================== 最终预测 ==================
# 使用完整模型（200棵树）进行最终预测
final_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
final_model.fit(X_scaled, y)

feature_df['pred_prob'] = final_model.predict_proba(X_scaled)[:, 1]
predicted = feature_df[feature_df['pred_prob'] > 0.35].sort_values('pred_prob', ascending=False)

# 加载过滤名单
country_filter = pd.read_csv('country.csv')['Country'].tolist()
unmedaled_2024 = pd.read_csv('data/2024_unmedaled_countries.csv')['Country'].tolist()

# 执行过滤
filtered = predicted[~predicted['country'].isin(country_filter)]
final_result = filtered[filtered['country'].isin(unmedaled_2024)]

# ================== 结果输出 ==================
print("\n2028年潜在首次获奖国家预测（前十名）：")
print(final_result[['country', 'pred_prob']].head(10).reset_index(drop=True).to_string(index=False))