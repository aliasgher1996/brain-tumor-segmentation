import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history_df, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    a1.plot(history_df['epoch'], history_df['loss'], label='train')
    a1.plot(history_df['epoch'], history_df['eval_loss'], label='validation')
    a1.set_ylabel('BCE loss')
    a1.legend(frameon=False)
    a2.plot(history_df['epoch'], history_df['dice_score'], label='train')
    a2.plot(history_df['epoch'], history_df['eval_dice_score'], label='validation')
    a2.set_ylabel('Dice score')
    a2.set_xlabel('Epoch')
    a2.legend(frameon=False)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, path, labels=('NORMAL', 'TUMOR')):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Greens')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=13,
                    color='white' if cm[i][j] > np.max(cm) / 2 else 'black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predictions(images, true_masks, pred_masks, path):
    """Rows of: MRI | MRI + true mask (green) | MRI + predicted mask (blue)."""
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for r, (img, t, p) in enumerate(zip(images, true_masks, pred_masks)):
        overlay_t = np.clip(img + np.dstack([t * 0.1, t * 0.45, t * 0.1]), 0, 1)
        overlay_p = np.clip(img + np.dstack([p * 0.3, p * 0.2, p * 0.8]), 0, 1)
        for ax, im, title in zip(axes[r], (img, overlay_t, overlay_p), ('MRI', 'True mask', 'Predicted mask')):
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
