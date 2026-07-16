"""Shared plotting helpers for baseline-model/svm_baseline.py and
CNN-models/train.py, so the two report results in a visually consistent,
directly comparable way.
"""

import matplotlib
matplotlib.use("Agg")  # headless: callers only save figures, never show them
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm, label_names, out_path):
    """Test-set confusion matrix as a heatmap, raw counts and row-normalized
    recall side by side -- counts alone hide class-imbalance effects (test
    partition sizes differ per class), row-normalization surfaces per-class
    recall directly."""
    cm = np.array(cm)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, matrix, title, fmt in [
        (axes[0], cm, "counts", "d"),
        (axes[1], cm_norm, "row-normalized (recall)", ".2f"),
    ]:
        ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(
            ax=ax, cmap="Blues", values_format=fmt, colorbar=False
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
