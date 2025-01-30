import matplotlib.pyplot as plt
import numpy as np

# Data for Chinese Women's Volleyball Team 2008
categories = ['Growth Rate', 'Medal Leap', 'Gold Breakthrough',
             'Medal Quality', 'Consistency', 'Competition', 'Event Weight']
values = [2.0, 0.25, 1.093, 0.833, 0.333, 0.725, 1.0]
weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]

# Data preprocessing
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

# Create radar chart
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Configure axes
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, fontsize=10)
ax.set_rlabel_position(0)
plt.yticks([0.5, 1.0, 1.5, 2.0], color="grey", size=8)
plt.ylim(0, 2.2)

# Plot weight baseline (dashed line)
weight_line = [w*8 for w in weights] + [weights[0]*8]
ax.plot(angles, weight_line, color='#1f77b4', linestyle='--',
        linewidth=1.5, label='Indicator Weights')

# Plot actual values (solid line)
ax.plot(angles, values, color='#d62728', linewidth=2.5,
        label='Actual Scores')
ax.fill(angles, values, color='#d62728', alpha=0.25)

# Add data labels
for angle, value, category in zip(angles[:-1], values[:-1], categories):
    plt.text(angle, value+0.1, f'{value:.2f}',
             ha='center', va='center',
             fontsize=9,
             rotation=np.degrees(angle)-90 if angle > np.pi else -np.degrees(angle))

# Add threshold circle
threshold_circle = [1.5]*len(angles)
ax.plot(angles, threshold_circle, color='#2ca02c',
        linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(np.pi/2, 1.6, 'Effect Threshold', color='#2ca02c',
        ha='center', va='bottom', fontsize=10)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
           fontsize=9)
plt.title('Radar Chart Analysis of Coaching Effect\nChinese Women\'s Volleyball Team (2008)',
          y=1.15, fontsize=12, fontweight='bold')

# Add annotation
plt.figtext(0.5, 0.02,
            "Note: Weight values amplified 8x for visualization",
            ha="center", fontsize=9, color='gray')

plt.tight_layout()
plt.show()