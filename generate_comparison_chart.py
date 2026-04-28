import matplotlib.pyplot as plt
import numpy as np

methods = ['Proposed\n(YOLOv8n)', 'Faster\nR-CNN', 'YOLOv3', 'SSD\nMobileNet', 'RetinaNet', 'Custom\nCNN']
mAP50 = [82.2, 77.8, 71.4, 68.2, 75.6, 68.2]
precision = [76.5, 74.2, 68.9, 64.5, 72.1, 65.4]
recall = [76.1, 72.5, 65.3, 61.8, 70.8, 63.7]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width, mAP50, width, label='mAP50 (%)', color='#1976D2')
bars2 = ax.bar(x, precision, width, label='Precision (%)', color='#388E3C')
bars3 = ax.bar(x + width, recall, width, label='Recall (%)', color='#F57C00')

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Comparative Analysis: GPR Object Detection Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc='upper right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Created: comparison_chart.png")