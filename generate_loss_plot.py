import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/detect/output/yolo/fine_k8_more30/results.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df['epoch'], df['train/box_loss'], 'b-o', linewidth=2, markersize=5, label='Train Box Loss')
axes[0].plot(df['epoch'], df['val/box_loss'], 'r--s', linewidth=2, markersize=5, label='Val Box Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Box Loss')
axes[0].set_title('Box Loss over Epochs')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['epoch'], df['train/cls_loss'], 'b-o', linewidth=2, markersize=5, label='Train Cls Loss')
axes[1].plot(df['epoch'], df['val/cls_loss'], 'r--s', linewidth=2, markersize=5, label='Val Cls Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Classification Loss')
axes[1].set_title('Classification Loss over Epochs')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('YOLOv8 Training and Validation Loss', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: loss_curves.png")