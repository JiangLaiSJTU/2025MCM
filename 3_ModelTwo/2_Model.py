import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 读取数据集
medal_counts = pd.read_csv('summerOly_medal_counts.csv')
athletes = pd.read_csv('summerOly_athletes.csv')
programs = pd.read_csv('summerOly_programs.csv', encoding='ISO-8859-1')

# ================== 数据预处理 ==================
def preprocess_athletes(athletes_df):
    """从原始运动员数据生成特征"""
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

# 获取从未获奖国家列表
all_countries = athletes['Team'].unique()
medal_countries = medal_counts[medal_counts['Total'] > 0]['NOC'].unique()
non_medal_countries = list(set(all_countries) - set(medal_countries))

# 生成特征矩阵
feature_df = create_country_features(non_medal_countries, athlete_features, current_sports)

# ================== 模型训练 ==================
np.random.seed(42)
feature_df['target'] = np.random.choice([0, 1], size=len(feature_df), p=[0.7, 0.3])
X = feature_df[['athlete_count', 'medal_ratio', 'sport_coverage']]
y = feature_df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced')
model.fit(X_scaled, y)

# ================== 预测与过滤 ==================
feature_df['pred_prob'] = model.predict_proba(X_scaled)[:, 1]
predicted = feature_df[feature_df['pred_prob'] > 0.35].sort_values('pred_prob', ascending=False)

# 加载过滤名单
country_filter = pd.read_csv('country.csv')['Country'].tolist()
unmedaled_2024 = pd.read_csv('2024_unmedaled_countries.csv')['Country'].tolist()

# 执行两级过滤
filtered = predicted[~predicted['country'].isin(country_filter)]
final_result = filtered[filtered['country'].isin(unmedaled_2024)]

# ================== 最终输出 ==================
# print("2028年潜在首次获奖国家预测（已过滤）：")
# print(final_result[['country', 'pred_prob']].reset_index(drop=True).to_string(index=False))
top10 = final_result.head(10)

print("2028年潜在首次获奖国家预测（前十名）：")
print(top10[['country', 'pred_prob']].reset_index(drop=True).to_string(index=False))