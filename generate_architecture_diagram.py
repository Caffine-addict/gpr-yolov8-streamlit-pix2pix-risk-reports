import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

colors = {
    'input': '#E3F2FD',
    'preprocess': '#FFF3E0',
    'detection': '#E8F5E9',
    'analysis': '#F3E5F5',
    'output': '#E0F7FA'
}
border_colors = {
    'input': '#1976D2',
    'preprocess': '#F57C00',
    'detection': '#388E3C',
    'analysis': '#7B1FA2',
    'output': '#00838F'
}

def draw_box(ax, x, y, w, h, text, color, border, fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor=border, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_arrow(ax, start, end, color='#455A64'):
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', color=color, lw=2))

ax.set_title('GPR B-Scan Analysis Pipeline Architecture', fontsize=14, fontweight='bold', pad=20)

draw_box(ax, 0.5, 6.5, 1.8, 1, 'Upload\nImage', colors['input'], border_colors['input'])
draw_box(ax, 3.5, 6.5, 1.8, 1, 'Load\nImage', colors['preprocess'], border_colors['preprocess'])
draw_box(ax, 6.5, 6.5, 1.8, 1, 'Median\nFilter', colors['preprocess'], border_colors['preprocess'])
draw_box(ax, 9.5, 6.5, 1.8, 1, 'CLAHE\nEnhance', colors['preprocess'], border_colors['preprocess'])

draw_box(ax, 0.5, 4.5, 1.8, 1, 'Denoised\nImage', colors['detection'], border_colors['detection'])
draw_box(ax, 3.5, 4.5, 1.8, 1, 'YOLO\nModel', colors['detection'], border_colors['detection'])
draw_box(ax, 6.5, 4.5, 1.8, 1, 'NMS &\nFilter', colors['detection'], border_colors['detection'])

draw_box(ax, 0.5, 2.5, 1.8, 1, 'Detection\nResults', colors['analysis'], border_colors['analysis'])
draw_box(ax, 3.5, 2.5, 1.8, 1, 'Gemma 4\nLLM', colors['analysis'], border_colors['analysis'])
draw_box(ax, 6.5, 2.5, 1.8, 1, 'Analysis\nReport', colors['analysis'], border_colors['analysis'])

draw_box(ax, 0.5, 0.3, 1.8, 1, 'Markdown\nReport', colors['output'], border_colors['output'])
draw_box(ax, 3.5, 0.3, 1.8, 1, 'Visualization\nOverlay', colors['output'], border_colors['output'])
draw_box(ax, 6.5, 0.3, 1.8, 1, 'Web UI\nDisplay', colors['output'], border_colors['output'])

draw_arrow(ax, (2.3, 7), (3.5, 7))
draw_arrow(ax, (5.3, 7), (6.5, 7))
draw_arrow(ax, (8.3, 7), (9.5, 7))

ax.annotate('', xy=(1.4, 6.5), xytext=(1.4, 5.5), arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

draw_arrow(ax, (2.3, 5), (3.5, 5))
draw_arrow(ax, (5.3, 5), (6.5, 5))

ax.annotate('', xy=(1.4, 4.5), xytext=(1.4, 3.5), arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

draw_arrow(ax, (2.3, 3), (3.5, 3))
draw_arrow(ax, (5.3, 3), (6.5, 3))

ax.annotate('', xy=(1.4, 2.5), xytext=(1.4, 1.3), arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))

draw_arrow(ax, (2.3, 0.8), (3.5, 0.8))
draw_arrow(ax, (5.3, 0.8), (6.5, 0.8))

ax.text(7.5, 7.3, 'PREPROCESSING', fontsize=10, fontweight='bold', color='#F57C00')
ax.text(7.5, 5.3, 'DETECTION', fontsize=10, fontweight='bold', color='#388E3C')
ax.text(7.5, 3.3, 'ANALYSIS', fontsize=10, fontweight='bold', color='#7B1FA2')
ax.text(7.5, 1.3, 'OUTPUT', fontsize=10, fontweight='bold', color='#00838F')

ax.text(11.5, 7, 'JPEG/PNG\n↓ numpy', fontsize=8, ha='center', color='#666')
ax.text(11.5, 5, '9 classes\nbounding boxes', fontsize=8, ha='center', color='#666')
ax.text(11.5, 3, 'Technical\nReport', fontsize=8, ha='center', color='#666')
ax.text(11.5, 1, 'Streamlit UI\nDisplay', fontsize=8, ha='center', color='#666')

plt.tight_layout()
plt.savefig('architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Created: architecture.png")