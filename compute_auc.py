"""
compute_auc.py — Tính AUC-ROC từ các file .npy anomaly scores đã lưu sau test_all_scenes.
Người phụ trách: Người 1 (ưu tiên xong cuối Ngày 2)

Usage:
    # STE-TTE
    python compute_auc.py --scores-dir experiments_andt_ADrone_STE_TTE/ \\
                          --labels-dir ../../UIT-ADrone/test/test_frame_mask/

    # Baseline CA
    python compute_auc.py --scores-dir experiments_andt_ADrone_baseline_CA/ \\
                          --labels-dir ../../UIT-ADrone/test/test_frame_mask/
"""

import argparse
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def compute_dataset_auc(scores_dir, labels_dir, plot=False, save_plot=None):
    """
    Tính AUC-ROC tổng hợp trên toàn bộ test set.

    Args:
        scores_dir: thư mục chứa file .npy anomaly scores (output của test_all_scenes)
        labels_dir: thư mục chứa file .npy ground-truth labels (0=normal, 1=anomaly)
        plot: vẽ ROC curve nếu True
        save_plot: đường dẫn lưu plot (nếu None thì show)

    Returns:
        float: AUC-ROC score
    """
    all_scores, all_labels = [], []

    score_files = sorted(glob.glob(os.path.join(scores_dir, '*.npy')))
    if not score_files:
        raise FileNotFoundError(f"Không tìm thấy file .npy nào trong {scores_dir}")

    for sf in score_files:
        name = os.path.basename(sf)
        label_path = os.path.join(labels_dir, name)
        if not os.path.exists(label_path):
            print(f"  [WARNING] Label not found for {name}, skipping.")
            continue

        scores = np.load(sf)
        labels = np.load(label_path, allow_pickle=True)

        # Align length (score có thể ngắn hơn label do drop_last=True)
        min_len = min(len(scores), len(labels))
        all_scores.extend(scores[:min_len].tolist())
        all_labels.extend(labels[len(labels) - min_len:].tolist())

        print(f"  {name}: {min_len} frames, anomaly ratio: {np.mean(labels[len(labels)-min_len:]):.2%}")

    if not all_labels:
        raise ValueError("Không có dữ liệu nào để tính AUC. Kiểm tra lại labels_dir.")

    auc = roc_auc_score(all_labels, all_scores)
    print(f"\nDataset AUC-ROC: {auc:.4f}")

    if plot:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve — {os.path.basename(scores_dir.rstrip("/"))}')
        plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot, dpi=150)
            print(f"ROC curve saved to {save_plot}")
        else:
            plt.show()

    return auc


def main():
    parser = argparse.ArgumentParser("Compute AUC-ROC from saved .npy anomaly score files")
    parser.add_argument("--scores-dir", type=str, required=True,
                        help="Thư mục chứa .npy anomaly scores (output của test_all_scenes)")
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Thư mục chứa .npy ground-truth labels")
    parser.add_argument("--plot", action='store_true', help="Vẽ ROC curve")
    parser.add_argument("--save-plot", type=str, default=None, help="Lưu ROC plot vào file")
    args = parser.parse_args()

    print(f"Scores dir : {args.scores_dir}")
    print(f"Labels dir : {args.labels_dir}")
    print()

    auc = compute_dataset_auc(
        scores_dir=args.scores_dir,
        labels_dir=args.labels_dir,
        plot=args.plot,
        save_plot=args.save_plot
    )
    return auc


if __name__ == '__main__':
    main()
