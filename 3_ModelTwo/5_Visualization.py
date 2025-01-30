import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import cm
import numpy as np
import pandas as pd


def plot_radar_chart(df):
    """极坐标雷达图展示国家潜力分布"""
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    categories = df['country'].tolist()
    values = df['pred_prob'].tolist()
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    ax.plot(angles, values, color='#FF6B6B', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='#FF6B6B', alpha=0.4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    plt.title('Olympic Medal Potential Radar Chart', y=1.1)
    plt.show()


def plot_interactive_bar(df):
    """交互式动态条形图（使用Plotly）"""
    fig = px.bar(df, x='pred_prob', y='country', orientation='h',
                 color='pred_prob', color_continuous_scale='Viridis',
                 title='2028 Olympic Medal Probability Forecast',
                 labels={'pred_prob': 'Probability (%)', 'country': 'Country'})
    fig.update_layout(height=500, width=800)
    fig.show()


def plot_gradient_bubble(df):
    """渐变气泡图表示潜力值"""
    plt.figure(figsize=(10, 6))
    norm = plt.Normalize(df['pred_prob'].min(), df['pred_prob'].max())
    colors = cm.viridis(norm(df['pred_prob']))

    scatter = plt.scatter(x=range(len(df)), y=df['pred_prob'],
                          s=df['pred_prob'] * 1000, c=colors, alpha=0.6)

    plt.xticks(range(len(df)), df['country'], rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title('Medal Probability Bubble Chart')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm).set_label('Probability Level')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_olympic_rings(df):
    """奥运五环主题水平条形图"""
    plt.figure(figsize=(10, 6))
    colors = ['#0085C7', '#000000', '#F4C300', '#009F3D', '#DF0024']

    bars = plt.barh(df['country'], df['pred_prob'],
                    color=colors * 2, edgecolor='white')

    # 添加奥运五环元素
    plt.text(0.5, 1.08, '⚫⚫⚫⚫⚫', transform=plt.gca().transAxes,
             fontsize=30, ha='center')

    plt.xlabel('Probability')
    plt.title('2028 Olympic Medal Predictions', pad=30)
    plt.xlim(0, 1)

    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + 0.2, f'{width:.1%}',
                 va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_medal_podium(df):
    """领奖台式可视化"""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))

    # 创建阶梯状领奖台
    podium = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]

    bars = plt.bar(x, df['pred_prob'],
                   color=plt.cm.viridis(np.linspace(0, 1, len(df))),
                   edgecolor='black')

    # 添加领奖台层次线
    for level in podium:
        plt.axhline(level, color='gray', linestyle='--', alpha=0.5)

    plt.xticks(x, df['country'], rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title('Olympic Medal Podium Prediction')
    plt.ylim(0, 1)

    # 添加金牌图标
    plt.text(x[0], df['pred_prob'][0] + 0.03, '🥇',
             fontsize=20, ha='center')
    plt.text(x[1], df['pred_prob'][1] + 0.03, '🥈',
             fontsize=20, ha='center')
    plt.text(x[2], df['pred_prob'][2] + 0.03, '🥉',
             fontsize=20, ha='center')

    plt.tight_layout()
    plt.show()


# 示例数据（使用您的预测结果）
data = {
    'country': ['Benin', 'Belize', 'Lesotho', 'Rwanda', 'Guinea',
                'DR Congo', 'Cambodia', 'Republic of Moldova',
                'Côte d\'Ivoire', 'Papua New Guinea'],
    'pred_prob': [0.965, 0.945, 0.875, 0.87, 0.795,
                  0.79, 0.76, 0.735, 0.727, 0.725]
}
df = pd.DataFrame(data)

# 调用可视化函数
plot_radar_chart(df)  # 极坐标雷达图
plot_interactive_bar(df)  # 交互式条形图（需Jupyter环境）
plot_gradient_bubble(df)  # 渐变气泡图
plot_olympic_rings(df)  # 奥运主题条形图
plot_medal_podium(df)  # 领奖台可视化