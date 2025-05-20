import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.ticker import MaxNLocator, ScalarFormatter,FixedLocator

# 设置全局字体样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def plot_tsne_from_saved(tsne_path='tsne_data.npz', seed=42):
    tsne_data = np.load(tsne_path)
    features = tsne_data['features']
    labels = tsne_data['labels']

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab10', len(unique_labels))

    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(features_2d[mask, 0],
                    features_2d[mask, 1],
                    color=cmap(label),
                    label=f'Type {label + 1}',
                    alpha=0.6,
                    edgecolors='w')

    ax = plt.gca()

    # 设置坐标轴范围
    plt.xlim(-40, 40)
    plt.ylim(-30, 30)

    # 设置刻度定位器
    ax.xaxis.set_major_locator(FixedLocator([-40, -20, 0, 20, 40]))
    ax.yaxis.set_major_locator(FixedLocator([-30, -15, 0, 15, 30]))

    # 设置刻度格式
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    ax.tick_params(axis='both', which='both',
                   bottom=True, top=True,
                   left=True, right=True,
                   direction='out',
                   labelsize=13)

    plt.title(f't-SNE Visualization', fontsize=13)
    plt.legend(title='Fault Types',
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               fontsize=10)
    plt.xlabel('Dimension 1', fontsize=13)
    plt.ylabel('Dimension 2', fontsize=13)

    plt.tight_layout()
    plt.savefig(f'tsne_full_ticks_seed_{seed}.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("t-SNE图已保存")


def plot_confusion_matrix_from_saved(cm_path='confusion_matrix.npy'):

    cm = np.load(cm_path)

    # 创建画布
    plt.figure(figsize=(8, 6))

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=np.arange(1, 7))  # 显示1-6
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')

    # 格式设置
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig('confusion_matrix_replot.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("混淆矩阵已保存")


if __name__ == "__main__":

    plot_tsne_from_saved(seed=42)
    plot_confusion_matrix_from_saved()
