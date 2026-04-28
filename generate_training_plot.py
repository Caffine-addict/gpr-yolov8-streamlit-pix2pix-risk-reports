import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/detect/output/yolo/fine_k8_more30/results.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(df['epoch'], df['metrics/precision(B)'], 'b-o', linewidth=2, markersize=5)
axes[0, 0].axvline(x=25, color='r', linestyle='--', alpha=0.7, label='Best Epoch (25)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].set_title('Precision over Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], 'g-o', linewidth=2, markersize=5)
axes[0, 1].axvline(x=25, color='r', linestyle='--', alpha=0.7, label='Best Epoch (25)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].set_title('Recall over Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], 'm-o', linewidth=2, markersize=5)
axes[1, 0].axvline(x=25, color='r', linestyle='--', alpha=0.7, label='Best Epoch (25)')
axes[1, 0].axhline(y=0.822, color='orange', linestyle=':', alpha=0.7, label='Best mAP50 (82.2%)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('mAP50')
axes[1, 0].set_title('mAP50 over Epochs')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], 'c-o', linewidth=2, markersize=5)
axes[1, 1].axvline(x=25, color='r', linestyle='--', alpha=0.7, label='Best Epoch (25)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('mAP50-95')
axes[1, 1].set_title('mAP50-95 over Epochs')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('YOLOv8 Training Metrics (30 Epochs)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: training_metrics.png")