"""Linear-SVM baseline on full-resolution, fully-preprocessed pixels.

Reuses src.dataset.SeasonDataset directly so the SVM sees exactly the same
crop/illumination-normalize/resize/tensor-normalize pipeline the CNN will,
isolating "does a learned representation beat a linear decision boundary" as
the variable under comparison (no resolution or preprocessing mismatch).

Raw pixels are 3*224*224=150,528-dim, which is impractically slow to fit
directly (both LinearSVC and SGDClassifier take 10+ minutes even on a few
hundred samples at that dimensionality). PCA reduces to a few hundred
components before the linear classifier -- the classic "eigenfaces"
approach -- while still being unsupervised (no hand-crafted colour
features), preserving the "raw pixels, no domain feature engineering"
framing this baseline is meant to have.

Usage:
    python baseline-model/svm_baseline.py [--label-mode {4,12}] [--limit N]
                                           [--model {linearsvc,sgd}]
                                           [--pca-components K]
                                           [--cv-folds K] [--n-jobs J] [--seed S]
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch.multiprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    """Flatten a Dataset yielding ((3,224,224) float tensor, scalar long label)
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
    n_features = 3 * config.IMG_SIZE * config.IMG_SIZE
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
    if dataset.label_mode == "4":
        return dataset.df["class"].map(config.CLASS_TO_IDX).to_numpy()
    return dataset.df.apply(
        lambda row: config.SUBCLASS_TO_IDX[(row["class"], row["sub_class"])], axis=1
    ).to_numpy()


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


def apply_pca(X_train, X_test, n_components, seed=42):
    """Fit PCA on X_train only, transform both -- an unsupervised, fixed
    (non-domain-specific) dimensionality reduction from 150,528 raw-pixel
    dims down to `n_components`, making the downstream linear classifier
    tractable. randomized solver: efficient for n_features >> n_components.
    """
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    return X_train_reduced, X_test_reduced, explained_variance


def get_label_names(label_mode):
    if label_mode == "4":
        return list(config.CLASSES)
    return [f"{c}-{s}" for c, s in config.SUBCLASS_COMBOS]


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
        # n_samples (~4000) < n_features (~150,528) -> dual=True is the
        # efficient liblinear regime; set explicitly rather than relying on
        # sklearn's version-dependent dual="auto".
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
    return GridSearchCV(base, grid, cv=cv, scoring="accuracy", n_jobs=n_jobs, refit=True)


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
    parser.add_argument("--label-mode", choices=["4", "12"], default="4")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use a stratified subsample of N rows per partition (fast smoke test)",
    )
    parser.add_argument("--model", choices=["linearsvc", "sgd"], default="linearsvc")
    parser.add_argument(
        "--pca-components", type=int, default=200,
        help="PCA components to reduce raw pixels to before the classifier (default: 200)",
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

    train_ds = SeasonDataset(partition="train", label_mode=args.label_mode)
    test_ds = SeasonDataset(partition="test", label_mode=args.label_mode)

    train_indices = stratified_limit_indices(train_ds, args.limit, seed=args.seed)
    test_indices = stratified_limit_indices(test_ds, args.limit, seed=args.seed)
    X_train, y_train = dataset_to_design_matrix(
        train_ds, indices=train_indices, num_workers=args.load_workers
    )
    X_test, y_test = dataset_to_design_matrix(
        test_ds, indices=test_indices, num_workers=args.load_workers
    )
    raw_train_shape, raw_test_shape = list(X_train.shape), list(X_test.shape)
    print(
        f"X_train {X_train.shape} ({X_train.nbytes / 1e9:.2f} GB), "
        f"X_test {X_test.shape} ({X_test.nbytes / 1e9:.2f} GB) "
        f"[loaded in {time.time() - t0:.1f}s with {args.load_workers} workers]"
    )

    t_pca = time.time()
    X_train, X_test, explained_variance = apply_pca(
        X_train, X_test, args.pca_components, seed=args.seed
    )
    print(
        f"PCA: {raw_train_shape[1]} -> {X_train.shape[1]} dims "
        f"(explained variance {explained_variance:.3f}) in {time.time() - t_pca:.1f}s"
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

    label_names = get_label_names(args.label_mode)
    results = {
        "model": args.model,
        "label_mode": args.label_mode,
        "best_params": search.best_params_,
        "cv_folds": safe_cv_folds(y_train, args.cv_folds),
        "raw_train_shape": raw_train_shape,
        "raw_test_shape": raw_test_shape,
        "pca_components": args.pca_components,
        "pca_explained_variance": explained_variance,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "elapsed_seconds": time.time() - t0,
        "convergence_warnings": convergence_warnings,
        **evaluate_model(search.best_estimator_, X_test, y_test, label_names),
    }

    out_dir = Path(__file__).resolve().parent
    out_name = (
        f"results_smoketest_{args.label_mode}.json"
        if args.limit
        else f"results_{args.label_mode}.json"
    )
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(results, indent=2))

    print(
        f"done in {results['elapsed_seconds']:.1f}s; "
        f"test accuracy = {results['accuracy']:.4f}; "
        f"best params = {search.best_params_}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
