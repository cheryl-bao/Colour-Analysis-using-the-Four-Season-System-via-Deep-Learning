"""Linear-SVM baseline on downsampled, fully-preprocessed pixels.

Reuses src.dataset.SeasonDataset directly so the SVM sees exactly the same
crop/illumination-normalize pipeline the CNN will -- just resized to a
smaller resolution (64x64 by default, via SeasonDataset's img_size, applied
as a tensor transform -- see src/dataset.py), isolating "does a learned
representation beat a linear decision boundary" as the variable under
comparison, modulo that one resolution difference.

Full-resolution (224x224 -> 150,528-dim) raw pixels are impractically slow
to fit directly (both LinearSVC and SGDClassifier take 10+ minutes even on
a few hundred samples at that dimensionality). At 64x64 -> 3*64*64=12,288 dims, that's no longer true:
the linear classifier fits directly on the flattened pixels, no
dimensionality reduction needed.

Usage:
    python baseline-model/svm_baseline.py [--limit N] [--model {linearsvc,sgd}]
                                           [--img-size N]
                                           [--cv-folds K] [--n-jobs J] [--seed S]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset

# DataLoader's default "file_descriptor" sharing strategy passes tensors
# between worker processes via shared memory / fd-passing, which crashes
# with a Bus error in this sandboxed environment (no /dev/shm). "file_system"
# avoids that mechanism, at the cost of more temp files under /tmp.
torch.multiprocessing.set_sharing_strategy("file_system")

from src import config
from src.dataset import SeasonDataset

LINEARSVC_C_GRID = [0.001, 0.01, 0.1, 1, 10]
SGD_ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def dataset_to_design_matrix(dataset, limit=None, indices=None, num_workers=0, batch_size=64):
    """Flatten a Dataset yielding ((C,H,W) float tensor, scalar long label)
    pairs into (X, y) numpy arrays. Pre-allocated to avoid a transient
    2x-memory list-then-stack. Loads via a DataLoader so `num_workers` > 0
    parallelizes the per-image decode/transform across processes -- default
    is 0 (single-process, deterministic) since that's what the unit tests
    exercise with tiny synthetic datasets; main() passes a higher value for
    real runs, where per-image PIL decode is otherwise the bottleneck.

    `indices`, if given, takes precedence over `limit` and selects exactly
    those dataset indices (in order, via torch.utils.data.Subset) -- used by
    main() to pass a stratified subsample for --limit smoke tests, since a
    naive head-N slice of annotations_processed.csv (grouped by class) can
    land on a single class.
    """
    if indices is not None:
        subset = Subset(dataset, list(int(i) for i in indices))
    elif limit is not None:
        subset = Subset(dataset, range(min(limit, len(dataset))))
    else:
        subset = dataset

    n = len(subset)
    # Inferred from an actual sample rather than a global constant, so this
    # works whatever resolution the passed dataset's transform resizes to
    # (main() may build datasets at different --img-size values).
    n_features = subset[0][0].numel() if n > 0 else 0
    X = np.empty((n, n_features), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pos = 0
    for images, labels in loader:
        batch_n = images.shape[0]
        X[pos : pos + batch_n] = images.reshape(batch_n, -1).numpy().astype(np.float32, copy=False)
        y[pos : pos + batch_n] = labels.numpy()
        pos += batch_n
    return X, y


def _season_dataset_labels(dataset):
    """Labels for a SeasonDataset without loading any images (dataset.df is
    already filtered to the dataset's partition; this mirrors __getitem__'s
    label derivation)."""
    return dataset.df["class"].map(config.CLASS_TO_IDX).to_numpy()


def stratified_limit_indices(dataset, limit, seed=42):
    """A stratified sample of `limit` indices from a SeasonDataset, so a
    --limit smoke test still spans multiple classes. Returns None if no
    subsampling is needed (limit is None or >= len(dataset))."""
    n_total = len(dataset)
    if limit is None or limit >= n_total:
        return None
    labels = _season_dataset_labels(dataset)
    indices, _ = train_test_split(
        np.arange(n_total), train_size=limit, stratify=labels, random_state=seed
    )
    return indices


def safe_cv_folds(y, requested_folds):
    """Clamp cv folds to the smallest class count so StratifiedKFold doesn't
    error out on tiny --limit smoke tests."""
    counts = np.bincount(y)
    min_class_count = counts[counts > 0].min()
    folds = max(2, min(requested_folds, min_class_count))
    if folds < requested_folds:
        print(
            f"warning: reducing cv folds {requested_folds} -> {folds} "
            f"(smallest class has only {min_class_count} samples)"
        )
    return folds


def build_search(model_name, y_train, cv_folds=5, seed=42, n_jobs=1):
    folds = safe_cv_folds(y_train, cv_folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    if model_name == "linearsvc":
        # n_samples (~4000) < n_features (~12,288 at the default 64x64 img
        # size) -> dual=True is the efficient liblinear regime; set
        # explicitly rather than relying on sklearn's version-dependent
        # dual="auto".
        base = LinearSVC(dual=True, max_iter=20000, random_state=seed)
        grid = {"C": LINEARSVC_C_GRID}
    elif model_name == "sgd":
        base = SGDClassifier(loss="hinge", max_iter=2000, tol=1e-3, random_state=seed)
        grid = {"alpha": SGD_ALPHA_GRID}
    else:
        raise ValueError(f"unknown model {model_name!r}")

    # n_jobs=1 by default: GridSearchCV parallelism is process-based (joblib
    # loky backend) and each worker holds its own copy of that fold's data.
    # Raising n_jobs trades memory for wall-clock speed -- not a default.
    # return_train_score=True costs extra fit time (train-fold scoring too)
    # but is what plot_validation_curve needs to show over/underfitting.
    return GridSearchCV(
        base, grid, cv=cv, scoring="accuracy", n_jobs=n_jobs, refit=True,
        return_train_score=True,
    )


def plot_validation_curve(search, model_name, out_path):
    """Train vs. CV accuracy across the searched regularization grid (C for
    LinearSVC, alpha for SGD) -- the SVM analogue of a CNN's per-epoch loss
    curve, since GridSearchCV has no notion of epochs. Reads mean/std
    train/test scores straight out of cv_results_ (no refitting)."""
    param_name = "C" if model_name == "linearsvc" else "alpha"
    results = search.cv_results_
    param_values = np.array([p[param_name] for p in results["params"]], dtype=float)
    order = np.argsort(param_values)

    param_values = param_values[order]
    train_mean, train_std = results["mean_train_score"][order], results["std_train_score"][order]
    test_mean, test_std = results["mean_test_score"][order], results["std_test_score"][order]

    fig, ax = plt.subplots(figsize=(6, 4))
    for mean, std, label in [(train_mean, train_std, "train"), (test_mean, test_std, "cv")]:
        ax.plot(param_values, mean, "o-", label=label)
        ax.fill_between(param_values, mean - std, mean + std, alpha=0.2)

    ax.set_xscale("log")
    ax.set_xlabel(param_name)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{model_name} validation curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


def evaluate_model(model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=list(range(len(label_names)))
        ).tolist(),
        "label_names": label_names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use a stratified subsample of N rows per partition (fast smoke test)",
    )
    parser.add_argument("--model", choices=["linearsvc", "sgd"], default="linearsvc")
    parser.add_argument(
        "--img-size", type=int, default=64,
        help="resize images to N x N (via SeasonDataset's tensor transform) before "
        "flattening -- small enough at the default to skip PCA entirely (default: 64)",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load-workers", type=int, default=min(8, os.cpu_count() or 1),
        help="parallel worker processes for loading/flattening images (default: min(8, cpu count))",
    )
    args = parser.parse_args()

    t0 = time.time()

    train_ds = SeasonDataset(partition="train", img_size=args.img_size)
    test_ds = SeasonDataset(partition="test", img_size=args.img_size)

    train_indices = stratified_limit_indices(train_ds, args.limit, seed=args.seed)
    test_indices = stratified_limit_indices(test_ds, args.limit, seed=args.seed)
    X_train, y_train = dataset_to_design_matrix(
        train_ds, indices=train_indices, num_workers=args.load_workers
    )
    X_test, y_test = dataset_to_design_matrix(
        test_ds, indices=test_indices, num_workers=args.load_workers
    )
    print(
        f"X_train {X_train.shape} ({X_train.nbytes / 1e9:.2f} GB), "
        f"X_test {X_test.shape} ({X_test.nbytes / 1e9:.2f} GB) "
        f"[loaded in {time.time() - t0:.1f}s with {args.load_workers} workers]"
    )

    search = build_search(
        args.model, y_train, cv_folds=args.cv_folds, seed=args.seed, n_jobs=args.n_jobs
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_train, y_train)
    convergence_warnings = [
        str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
    ]
    if convergence_warnings:
        suggestion = (
            "consider raising max_iter or using --model sgd"
            if args.model == "linearsvc"
            else "consider raising max_iter"
        )
        print(
            f"warning: {len(convergence_warnings)} ConvergenceWarning(s) during search; {suggestion}"
        )

    label_names = [config.CLASS_DISPLAY_NAMES[c] for c in config.CLASSES]
    results = {
        "model": args.model,
        "img_size": args.img_size,
        "best_params": search.best_params_,
        "cv_folds": safe_cv_folds(y_train, args.cv_folds),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "elapsed_seconds": time.time() - t0,
        "convergence_warnings": convergence_warnings,
        **evaluate_model(search.best_estimator_, X_test, y_test, label_names),
    }

    out_dir = Path(__file__).resolve().parent
    suffix = "smoketest" if args.limit else None
    out_path = out_dir / (f"results_{suffix}.json" if suffix else "results.json")
    curve_path = out_dir / (f"validation_curve_{suffix}.png" if suffix else "validation_curve.png")
    cm_path = out_dir / (f"confusion_matrix_{suffix}.png" if suffix else "confusion_matrix.png")

    plot_validation_curve(search, args.model, curve_path)
    results["validation_curve_path"] = curve_path.name

    plot_confusion_matrix(results["confusion_matrix"], label_names, cm_path)
    results["confusion_matrix_path"] = cm_path.name

    out_path.write_text(json.dumps(results, indent=2))

    print(
        f"done in {results['elapsed_seconds']:.1f}s; "
        f"test accuracy = {results['accuracy']:.4f}; "
        f"best params = {search.best_params_}"
    )
    print(f"wrote {out_path}")
    print(f"wrote {curve_path}")
    print(f"wrote {cm_path}")


if __name__ == "__main__":
    main()
