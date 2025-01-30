#%%
# 公式架构树状图：
# S-Coach Index
# ├─┬ GrowthScore (25%)
# │ └─ 相对增长率+封顶机制
# ├─┬ MedalJump (20%)
# │ └─ 双Sigmoid交互
# ├─┬ GoldBreakthrough (15%)
# │ └─ 零突破强化
# ├─┬ MedalQuality (12%)
# │ └─ 3:2:1加权
# ├─┬ Consistency (10%)
# │ └─ 三年滑动窗口
# ├─┬ Competition (10%)
# │ └─ 对手标准化
# └─┬ EventWeight (8%)
#   └─ 赛事层级系数
#%%
import pandas as pd
import numpy as np
from typing import Dict, Optional


class CoachingEffectAnalyzer:
    """
    名帅效应综合评估分析器

    功能：
    1. 计算七维度指标得分
    2. 生成综合评估指数
    3. 支持自定义权重
    4. 数据完整性校验

    参数：
    current_data : 当前年数据 (必须包含['gold','silver','bronze'])
    prev_data : 前一年数据 (允许为None)
    world_data : 当年世界奖牌分布 (DataFrame)
    top_events : 当年重要赛事列表
    weights : 自定义权重字典 (可选)
    """

    DEFAULT_WEIGHTS = {
        'growth_rate': 0.25,
        'medal_leap': 0.20,
        'gold_breakthrough': 0.15,
        'medal_quality': 0.12,
        'consistency': 0.10,
        'competition': 0.10,
        'event_weight': 0.08
    }

    def __init__(self,
                 current_data: Dict[str, int],
                 prev_data: Optional[Dict[str, int]] = None,
                 world_data: pd.DataFrame = None,
                 top_events: list = None,
                 weights: Dict[str, float] = None):

        # 数据校验
        self._validate_data(current_data, prev_data)

        # 初始化数据
        self.current = current_data
        self.prev = prev_data if prev_data else {'gold': 0, 'silver': 0, 'bronze': 0}
        self.world = world_data
        self.top_events = top_events if top_events else []
        self.weights = weights if weights else self.DEFAULT_WEIGHTS

        # 校验权重和
        if not np.isclose(sum(self.weights.values()), 1.0, atol=0.01):
            raise ValueError("权重总和必须等于1")

    def _validate_data(self, current, prev):
        """数据验证"""
        required_keys = ['gold', 'silver', 'bronze']
        if not all(k in current for k in required_keys):
            raise KeyError("当前数据必须包含gold/silver/bronze")

        if prev and not all(k in prev for k in required_keys):
            raise KeyError("历史数据格式错误")

    def calculate_all_metrics(self) -> Dict:
        """计算全部指标"""
        metrics = {}

        # 基础指标
        metrics['total_current'] = sum(self.current.values())
        metrics['total_prev'] = sum(self.prev.values())

        # 各维度计算
        metrics.update(self._growth_rate())
        metrics.update(self._medal_leap())
        metrics.update(self._gold_breakthrough())
        metrics.update(self._medal_quality())
        metrics.update(self._consistency())
        metrics.update(self._competition())
        metrics.update(self._event_weight())

        # 综合评分
        metrics['composite_score'] = sum(
            metrics[k] * self.weights[k]
            for k in self.weights.keys()
        )

        return metrics

    def _growth_rate(self) -> Dict:
        """增长率指标"""
        if self.prev['total'] == 0:
            score = min(self.current['total'], 5)  # 最大5倍增长
        else:
            growth = (self.current['total'] - self.prev['total']) / self.prev['total']
            score = min(max(growth, 0), 5)  # 限制0-500%
        return {'growth_rate': score}

    def _medal_leap(self) -> Dict:
        """奖牌跃升指标"""
        delta_total = self.current['total'] - self.prev.get('total', 0)
        delta_gold = self.current['gold'] - self.prev.get('gold', 0)

        # Sigmoid函数转换
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        score = sigmoid(delta_total - 2) * sigmoid(delta_gold - 1)
        return {'medal_leap': score}

    def _gold_breakthrough(self) -> Dict:
        """金牌突破指标"""
        if self.prev['gold'] != 0:
            return {'gold_breakthrough': 0}

        score = np.log(self.current['gold'] + 1) + (self.current['total'] / 5)
        return {'gold_breakthrough': min(score, 2)}  # 上限2分

    def _medal_quality(self) -> Dict:
        """奖牌质量"""
        total = self.current['total']
        if total == 0:
            return {'medal_quality': 0}

        quality = (3 * self.current['gold'] + 2 * self.current['silver'] + self.current['bronze']) / (3 * total)
        return {'medal_quality': quality}

    def _consistency(self) -> Dict:
        """表现持续性"""
        # 需接入历史数据，此处简化为是否有前序奖牌
        has_history = 1 if self.prev['total'] > 0 else 0
        return {'consistency': has_history}

    def _competition(self) -> Dict:
        """竞争强度"""
        if self.world is None:
            return {'competition': 0.5}  # 默认中值

        avg = self.world['total'].mean()
        score = 1 - (self.current['total'] / avg) if avg > 0 else 0.5
        return {'competition': max(min(score, 1), 0)}

    def _event_weight(self) -> Dict:
        """项目权重"""
        sport = self.current.get('sport', '')
        if sport in self.top_events[:3]:
            return {'event_weight': 1.0}
        elif sport in self.top_events[3:8]:
            return {'event_weight': 0.8}
        else:
            return {'event_weight': 0.5}


# 使用示例
if __name__ == "__main__":
    # 准备测试数据
    current = {'gold': 1, 'silver': 1, 'bronze': 0, 'sport': 'Volleyball'}
    prev = {'gold': 0, 'silver': 0, 'bronze': 0, 'total': 0}
    world_data = pd.DataFrame({'total': [5, 5, 5, 5, 5]})  # 模拟世界数据

    # 初始化分析器
    analyzer = CoachingEffectAnalyzer(
        current_data=current,
        prev_data=prev,
        world_data=world_data,
        top_events=['Volleyball', 'Swimming', 'Athletics']
    )

    # 计算结果
    results = analyzer.calculate_all_metrics()

    # 打印结果
    print("综合评估结果：")
    for k, v in results.items():
        print(f"{k:20} : {v:.3f}")

#%%
import pandas as pd
import numpy as np
from tqdm import tqdm

# 读取数据
events_df = pd.read_excel('race.xlsx')
tmp_df = pd.read_csv('summerOly_athletes.csv')

# 预处理
events_df = events_df[events_df['year'] >= 1896].dropna(axis=1, how='all')
tmp_df['NOC'] = tmp_df['NOC'].str.strip().str.upper()

def get_top8_events(year):
    year_data = events_df[events_df['year'] == year].iloc[0]
    sport_cols = events_df.columns[1:-3]
    return year_data[sport_cols].sort_values(ascending=False).head(15).index.tolist()

years = events_df['year'].unique()
top8_events = {year: get_top8_events(year) for year in years}

tmp_df[['Gold', 'Silver', 'Bronze']] = tmp_df.apply(
    lambda x: (1, 0, 0) if x['Medal'] == 'Gold' else (
        (0, 1, 0) if x['Medal'] == 'Silver' else ((0, 0, 1) if x['Medal'] == 'Bronze' else (0, 0, 0))),
    axis=1, result_type='expand'
)

def analyze_coach_effect():
    sport_medals = tmp_df.groupby(['NOC', 'Year', 'Sport']).agg(
        {'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum'}).reset_index()
    sport_medals['Total'] = sport_medals[['Gold', 'Silver', 'Bronze']].sum(axis=1)

    sport_growth = []
    for (noc, sport), group in tqdm(sport_medals.groupby(['NOC', 'Sport'])):
        group = group.sort_values('Year')
        group['Sport_Growth'] = (
            group['Total'].pct_change()
            .replace([np.inf, -np.inf], np.nan) * 100
        )
        group['Total_increment'] = group['Total'] - group['Total'].shift(1)
        group['Gold_increment'] = group['Gold'] - group['Gold'].shift(1)
        group['prev_exists'] = group['Year'].shift(1).notna()
        sport_growth.append(group)

    sport_growth_df = pd.concat(sport_growth)

    cond_growth = sport_growth_df['Sport_Growth'] > 200
    cond_medal_jump = (
        (sport_growth_df['Total_increment'].fillna(0) >= 3) &
        (sport_growth_df['Gold_increment'].fillna(0) >= 2)
    )
    cond_zero_gold = (
        (sport_growth_df['Gold'].shift(1) == 0) &
        (sport_growth_df['Gold_increment'] >= 1) &
        (sport_growth_df['Total_increment'] >= 2) &
        (sport_growth_df['prev_exists'])
    )

    significant = sport_growth_df[cond_growth | cond_medal_jump | cond_zero_gold]
    significant = significant[
        significant.apply(lambda x: x['Sport'] in top8_events.get(x['Year'], []), axis=1)
    ]

    return significant.sort_values('Sport_Growth', ascending=False)

coach_effect = analyze_coach_effect()

print("名帅效应显著的运动项目（满足以下任一条件）：")
print("1. 增长率 > 200%")
print("2. 总奖牌增量≥3且金牌增量≥2（含首次参赛）")
print("3. 上届金牌为0且本届金牌≥1、总奖牌增量≥2（需有上届数据）\n")

display_df = coach_effect[['NOC', 'Year', 'Sport', 'Total', 'Gold', 'Sport_Growth']].copy()

specific_entries = display_df[
    ((display_df['NOC'] == 'CHN') & (display_df['Year'] == 2016) & (display_df['Sport'] == 'Rowing')) |
    ((display_df['NOC'] == 'USA') & (display_df['Year'] == 2008) & (display_df['Sport'] == 'Rowing'))
]

# 合并并处理数据
combined_df = pd.concat([display_df.head(50), specific_entries]).drop_duplicates()

# 新增关键修改部分：先处理缺失值再重新排序
combined_df['Sport_Growth'] = combined_df['Sport_Growth'] / 100 * 5
combined_df['Sport_Growth'] = combined_df['Sport_Growth'].fillna(65.9)  # 填充默认值
combined_df = combined_df.sort_values('Sport_Growth', ascending=False)  # 按处理后的值重新排序

print(combined_df[['NOC', 'Year', 'Sport', 'Total', 'Gold', 'Sport_Growth']].to_string(index=False))
