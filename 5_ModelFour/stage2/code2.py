# import csv
#
# # 定义数据
# data = [
#     ["Indicator", "Original Weight", "New Weight", "Adjustment Description"],
#     ["Growth Rate Indicator (F1)", "0.25", "0.22", "Slightly Reduced"],
#     ["Medal Leap (F2)", "0.2", "0.18", "Slightly Reduced"],
#     ["Gold Breakthrough (F3)", "0.15", "0.14", "Slightly Reduced"],
#     ["Medal Quality (F4)", "0.12", "0.11", "Slightly Reduced"],
#     ["Performance Consistency (F5)", "0.1", "0.09", "Slightly Reduced"],
#     ["Competitiveness (F6)", "0.1", "0.09", "Slightly Reduced"],
#     ["Event Weight (F7)", "0.08", "0.07", "Slightly Reduced"],
#     ["Athlete Stability (F8)", "-", "0.1", "New Indicator"]
# ]
#
# # 指定CSV文件名
# filename = 'indicators_adjustment.csv'
#
# # 写入CSV文件
# with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(data)
#
# print(f"数据已保存到 {filename}")

import matplotlib.pyplot as plt

# Data from the image
data = {
    'Indicator': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
    'Score': [0.667, 0.516, 1.000, 0.667, 0.762, 0.500, 1.000, 0.325],
    'Weight': [0.22, 0.18, 0.10, 0.11, 0.18, 0.09, 0.08, 0.04]
}

# Indicator names in English
indicator_names = {
    'F1': 'Growth Rate Indicator (F1)',
    'F2': 'Medal Leap (F2)',
    'F3': 'Gold Breakthrough (F3)',
    'F4': 'Medal Quality (F4)',
    'F5': 'Performance Consistency (F5)',
    'F6': 'Competitiveness (F6)',
    'F7': 'Event Weight (F7)',
    'F8': 'Athlete Stability (F8)'
}

# Calculate weighted scores
weighted_scores = [score * weight for score, weight in zip(data['Score'], data['Weight'])]

# Set the style of the plot
plt.style.use('seaborn-v0_8-dark')

# Create a pie chart
plt.figure(figsize=(10, 8))

# Calculate the percentage and absolute values for the pie chart
percentages = [weight for weight in data['Weight']]
absolutes = weighted_scores

# Create the pie chart with percentage and absolute values
plt.pie(percentages, labels=[indicator_names[i] for i in data['Indicator']],
        autopct=lambda p: f'{p:.1f}%\n({absolutes[int(8 * p / 100)]:.4f})', startangle=140,
        textprops={'fontsize': 10})

# Add a legend
plt.legend(data['Indicator'], title="Indicators", loc="lower right", bbox_to_anchor=(1.1, 0),
           title_fontsize='small', fontsize='small')

# Add a title
plt.title('Weight Distribution and Weighted Scores of Indicators', fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()