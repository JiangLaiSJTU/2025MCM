# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Data from the image
# data = {
#     'Indicator': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
#     'Gold Medal Score': [1, 0.4523, 0.8931, 1, 0.45, 0.5, 1, 0.6985],
#     'Weight': [0.22, 0.18, 0.14, 0.11, 0.09, 0.09, 0.07, 0.10]
# }
#
# # Create a DataFrame
# df = pd.DataFrame(data)
#
# # Calculate the weighted score
# df['Weighted Score'] = df['Gold Medal Score'] * df['Weight']
#
# def plot_bar_chart():
#     # Set the style of the plot
#     sns.set(style="whitegrid")
#
#     # Create a bar plot
#     plt.figure(figsize=(10, 6))
#     barplot = sns.barplot(x='Indicator', y='Weighted Score', data=df, palette='coolwarm')
#
#     # Add labels and title
#     plt.xlabel('Indicator', fontsize=12)
#     plt.ylabel('Weighted Score', fontsize=12)
#     plt.title('Weighted Scores of Indicators Based on Gold Medals', fontsize=14)
#
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
# def plot_scatter_chart():
#     # Create a scatter plot
#     plt.figure(figsize=(10, 6))
#     scatterplot = sns.scatterplot(x='Gold Medal Score', y='Weighted Score', size='Weight', data=df, legend=False, sizes=(20, 200), palette='muted')
#
#     # Add labels and title
#     plt.xlabel('Gold Medal Score', fontsize=12)
#     plt.ylabel('Weighted Score', fontsize=12)
#     plt.title('Weighted Scores vs Gold Medal Scores', fontsize=14)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
# def plot_heatmap():
#     # Transpose the dataframe for heatmap
#     df_transposed = df.set_index('Indicator').T
#
#     # Create a heatmap
#     plt.figure(figsize=(8, 6))
#     heatmap = sns.heatmap(df_transposed, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Values'})
#
#     # Add title
#     plt.title('Heatmap of Indicators Scores and Weights', fontsize=14)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
# # Call the functions to plot the charts
# plot_bar_chart()
# plot_scatter_chart()
# plot_heatmap()

#
# import matplotlib.pyplot as plt
#
# # Data from the image
# data = {
#     'Indicator': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
#     'Weight': [0.22, 0.18, 0.14, 0.11, 0.09, 0.09, 0.07, 0.10],
#     'Gold Medal Score': [1, 0.4523, 0.8931, 1, 0.45, 0.5, 1, 0.6985],
#     'Weighted Score': [0.22, 0.08141, 0.125034, 0.11, 0.0405, 0.045, 0.07, 0.06985]
# }
#
# # Indicator names in English
# indicator_names = {
#     'F1': 'Growth Rate (F1)',
#     'F2': 'Medal Leap (F2)',
#     'F3': 'Gold Breakthrough (F3)',
#     'F4': 'Medal Quality (F4)',
#     'F5': 'Performance Consistency (F5)',
#     'F6': 'Competition Intensity (F6)',
#     'F7': 'Project Weight (F7)',
#     'F8': 'Athlete Stability (F8)'
# }
#
# def plot_pie_chart():
#     # Set the style of the plot
#     plt.style.use('seaborn-v0_8-dark')
#
#     # Create a pie chart
#     plt.figure(figsize=(8, 8))
#     plt.pie(data['Weight'], labels=[indicator_names[i] for i in data['Indicator']],
#             autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
#
#     # Add a legend
#     plt.legend(data['Indicator'], title="Indicators", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
#
#     # Add a title
#     plt.title('Weight Distribution of Indicators', fontsize=14)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
# # Call the function to plot the pie chart
# plot_pie_chart()

# import matplotlib.pyplot as plt
#
# # Data from the image
# data = {
#     'Indicator': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
#     'Weight': [0.22, 0.18, 0.14, 0.11, 0.09, 0.09, 0.07, 0.10],
#     'Gold Medal Score': [1, 0.4523, 0.8931, 1, 0.45, 0.5, 1, 0.6985],
#     'Weighted Score': [0.22, 0.08141, 0.125034, 0.11, 0.0405, 0.045, 0.07, 0.06985]
# }
#
# # Indicator names in English
# indicator_names = {
#     'F1': 'Growth Rate (F1)',
#     'F2': 'Medal Leap (F2)',
#     'F3': 'Gold Breakthrough (F3)',
#     'F4': 'Medal Quality (F4)',
#     'F5': 'Performance Consistency (F5)',
#     'F6': 'Competition Intensity (F6)',
#     'F7': 'Project Weight (F7)',
#     'F8': 'Athlete Stability (F8)'
# }
#
#
# def plot_pie_chart():
#     # Set the style of the plot
#     plt.style.use('seaborn-v0_8-dark')
#
#     # Create a pie chart
#     plt.figure(figsize=(10, 8))
#
#     # Calculate the percentage and absolute values for the pie chart
#     percentages = data['Weight']
#     absolutes = data['Weighted Score']
#
#     # Create the pie chart with percentage and absolute values
#     wedges, texts, autotexts = plt.pie(percentages, labels=[indicator_names[i] for i in data['Indicator']],
#                                        autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
#
#     # Add the absolute values to the pie chart
#     for i, p in enumerate(autotexts):
#         p.set_text(f"{p.get_text()}\n({absolutes[i]:.4f})")
#
#     # Add a legend
#     plt.legend(data['Indicator'], title="Indicators", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
#
#     # Add a title
#     plt.title('Weight Distribution and Weighted Scores of Indicators', fontsize=14)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
#
# # Call the function to plot the pie chart
# plot_pie_chart()

# import matplotlib.pyplot as plt
#
# # Data from the image
# data = {
#     'Indicator': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
#     'Weight': [0.22, 0.18, 0.14, 0.11, 0.09, 0.09, 0.07, 0.10],
#     'Weighted Score': [0.22, 0.08141, 0.125034, 0.11, 0.0405, 0.045, 0.07, 0.06985]
# }
#
# # Indicator names in English
# indicator_names = {
#     'F1': 'Growth Rate (F1)',
#     'F2': 'Medal Leap (F2)',
#     'F3': 'Gold Breakthrough (F3)',
#     'F4': 'Medal Quality (F4)',
#     'F5': 'Performance Consistency (F5)',
#     'F6': 'Competition Intensity (F6)',
#     'F7': 'Project Weight (F7)',
#     'F8': 'Athlete Stability (F8)'
# }
#
#
# def plot_pie_chart():
#     # Set the style of the plot
#     plt.style.use('seaborn-v0_8-dark')
#
#     # Create a pie chart
#     plt.figure(figsize=(10, 8))
#
#     # Calculate the percentage and absolute values for the pie chart
#     percentages = data['Weight']
#     absolutes = data['Weighted Score']
#
#     # Create the pie chart with percentage and absolute values
#     plt.pie(percentages, labels=[indicator_names[i] for i in data['Indicator']],
#             autopct=lambda p: f'{p:.1f}%\n({absolutes[int(8 * p / 100)]:.4f})', startangle=140,
#             textprops={'fontsize': 10})
#
#     # Add a legend
#     plt.legend(data['Indicator'], title="Indicators", loc="lower right", bbox_to_anchor=(1.1, 0),
#                title_fontsize='small', fontsize='small')
#
#     # Add a title
#     plt.title('Weight Distribution and Weighted Scores of Indicators', fontsize=14)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
#
# # Call the function to plot the pie chart
# plot_pie_chart()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建数据
data = {
    'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    '2016': ['Mashu Baker', 'Masashi Ebinuma', 'Ryunosuke Haga', 'Hisayoshi Harasawa', 'Ami Kondo',
             'Kaori Matsumoto', 'Takanori Nagase', 'Misato Nakamura', 'Shohei Ono',
             'Haruka Tachimoto', 'Naohisa Takato', 'Miku Tashiro', 'Mami Umeki', 'Kanae Yamabe'],
    '2020': ['Hifumi Abe', 'Uta Abe', 'Chizuru Arai', 'Shori Hamada', 'Hisayoshi Harasawa',
             'Shoichiro Mukai', 'Takanori Nagase', 'Shohei Ono', 'Akira Sone',
             'Naohisa Takato', 'Miku Tashiro', 'Funa Tonaki', 'Aaron Wolf', 'Tsukasa Yoshida'],
    '2024': ['Haruka Funakubo', 'Natsumi Tsunoda', 'Soichi Hashimoto', 'Rika Takayama', 'Saki Niizoe',
             'Tatsuru Saito', 'Sanshiro Murao', 'Ryuju Nagayama', 'Uta Abe',
             'Akira Sone', 'Hifumi Abe', 'Aaron Wolf', 'Takanori Nagase', 'Miku Takaichi']
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 设置绘图风格
sns.set(style="whitegrid")

# 创建热图
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df.set_index('Rank'), annot=True, fmt="s", cmap="YlGnBu", cbar=False)

# 设置标题和标签
plt.title('Judo Athletes by Year and Rank')
plt.xlabel('Year')
plt.ylabel('Rank')

# 显示图形
plt.show()