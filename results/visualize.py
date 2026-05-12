"""
results/visualize.py — Visualization utilities cho anomaly detection
Người phụ trách: Người 3 (chuẩn bị Ngày 2, chạy Ngày 3)

Usage (sau khi đã có .npy score files từ cả 2 model):
    python results/visualize.py
"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.makedirs('results', exist_ok=True)


# ─── ROC Curve Comparison ──────────────────────────────────────────────────────

def plot_roc_comparison(models: dict, labels, save_path='results/roc_comparison.png'):
    """
    Vẽ ROC curve của nhiều model trên cùng 1 plot.

    Args:
        models: dict {'Model Name': scores_array}
        labels: ground-truth binary labels array
        save_path: đường dẫn lưu ảnh
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, scores in models.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — UIT-ADrone Anomaly Detection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Saved: {save_path}')
    plt.close()


# ─── Anomaly Score Timeline ────────────────────────────────────────────────────

def plot_score_timeline(scene_name, scores_dir, labels_dir, out_dir='results/'):
    """
    Vẽ anomaly score theo thời gian cho 1 video scene.

    Args:
        scene_name: tên scene (ví dụ: 'DJI_0073')
        scores_dir: thư mục chứa .npy scores
        labels_dir: thư mục chứa .npy labels
        out_dir: thư mục lưu ảnh output
    """
    scores = np.load(os.path.join(scores_dir, scene_name + '.npy'))
    labels = np.load(os.path.join(labels_dir, scene_name + '.npy'), allow_pickle=True)

    # Align length
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[len(labels) - min_len:]

    plt.figure(figsize=(14, 4))
    plt.plot(scores, label='Anomaly Score (MSE)', color='steelblue', linewidth=1)
    plt.fill_between(range(len(labels)), 0, scores.max(),
                     where=(labels == 1), alpha=0.3, color='red', label='Ground Truth Anomaly')

    threshold = np.mean(scores) + np.std(scores)
    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=1.5,
                label=f'Threshold (mean+std) = {threshold:.4f}')

    plt.xlabel('Frame Index')
    plt.ylabel('MSE Loss')
    plt.title(f'Anomaly Detection Score — {scene_name}')
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{scene_name}_scores.png')
    plt.savefig(save_path, dpi=120)
    print(f'Saved: {save_path}')
    plt.close()


# ─── Load all scores from a directory ─────────────────────────────────────────

def load_all_scores(scores_dir, labels_dir):
    """Load và align toàn bộ scores + labels từ một thư mục."""
    all_scores, all_labels = [], []
    score_files = sorted(glob.glob(os.path.join(scores_dir, '*.npy')))
    for sf in score_files:
        name = os.path.basename(sf)
        label_path = os.path.join(labels_dir, name)
        if not os.path.exists(label_path):
            continue
        scores = np.load(sf)
        labels = np.load(label_path, allow_pickle=True)
        min_len = min(len(scores), len(labels))
        all_scores.extend(scores[:min_len].tolist())
        all_labels.extend(labels[len(labels) - min_len:].tolist())
    return np.array(all_scores), np.array(all_labels)


# ─── Main: chạy sau khi có đủ score files từ cả 2 model ─────────────────────

if __name__ == '__main__':
    LABELS_DIR  = '../../UIT-ADrone/test/test_frame_mask/'
    STE_TTE_DIR = '../experiments_andt_ADrone_STE_TTE/'
    BASELINE_DIR = '../experiments_andt_ADrone_baseline_CA/'

    # --- Load scores ---
    ste_scores, labels = load_all_scores(STE_TTE_DIR, LABELS_DIR)
    baseline_scores, _ = load_all_scores(BASELINE_DIR, LABELS_DIR)

    # --- Plot ROC comparison ---
    plot_roc_comparison(
        models={
            'Baseline CA':        baseline_scores,
            'STE-TTE (Proposed)': ste_scores,
        },
        labels=labels,
        save_path='results/roc_comparison.png'
    )

    # --- Plot score timeline cho từng scene ---
    scenes = [os.path.splitext(os.path.basename(f))[0]
              for f in glob.glob(os.path.join(STE_TTE_DIR, '*.npy'))]
    for scene in scenes:
        try:
            plot_score_timeline(scene, STE_TTE_DIR, LABELS_DIR, out_dir='results/scores/')
        except Exception as e:
            print(f"[WARN] Skip {scene}: {e}")

    print("\nDone! Visualizations saved to results/")
