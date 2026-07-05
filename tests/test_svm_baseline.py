import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODULE_PATH = Path(__file__).resolve().parent.parent / "baseline-model" / "svm_baseline.py"
spec = importlib.util.spec_from_file_location("svm_baseline", MODULE_PATH)
svm_baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svm_baseline)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.full((3, 224, 224), float(idx), dtype=torch.float32)
        return image, torch.tensor(idx % 4, dtype=torch.long)


def test_dataset_to_design_matrix_shape_and_dtype():
    X, y = svm_baseline.dataset_to_design_matrix(_TinyDataset(6))
    assert X.shape == (6, 3 * 224 * 224)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert (X[3] == 3.0).all()


def test_dataset_to_design_matrix_respects_limit():
    X, y = svm_baseline.dataset_to_design_matrix(_TinyDataset(10), limit=3)
    assert X.shape[0] == 3 and y.shape[0] == 3


def test_dataset_to_design_matrix_respects_indices():
    X, y = svm_baseline.dataset_to_design_matrix(_TinyDataset(10), indices=[7, 2])
    assert X.shape[0] == 2
    assert (X[0] == 7.0).all() and y[0] == 7 % 4
    assert (X[1] == 2.0).all() and y[1] == 2 % 4


class _FakeSeasonDataset:
    """Mimics the .df/.label_mode surface stratified_limit_indices needs,
    without depending on real data/processed/ files."""

    def __init__(self, classes, label_mode="4"):
        self.df = pd.DataFrame({"class": classes, "sub_class": ["deep"] * len(classes)})
        self.label_mode = label_mode

    def __len__(self):
        return len(self.df)


def test_stratified_limit_indices_spans_all_classes():
    # 40 rows grouped by class, like the real (sorted) annotations CSV --
    # a naive head-N slice would land on a single class.
    classes = ["autunno"] * 10 + ["estate"] * 10 + ["inverno"] * 10 + ["primavera"] * 10
    ds = _FakeSeasonDataset(classes)

    indices = svm_baseline.stratified_limit_indices(ds, limit=8, seed=0)

    assert indices is not None
    assert len(indices) == 8
    sampled_classes = {classes[i] for i in indices}
    assert sampled_classes == {"autunno", "estate", "inverno", "primavera"}


def test_stratified_limit_indices_none_when_limit_not_smaller():
    ds = _FakeSeasonDataset(["autunno", "estate"])
    assert svm_baseline.stratified_limit_indices(ds, limit=None) is None
    assert svm_baseline.stratified_limit_indices(ds, limit=2) is None


def test_apply_pca_reduces_dimensionality():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 500)).astype(np.float32)
    X_test = rng.normal(size=(10, 500)).astype(np.float32)

    X_train_reduced, X_test_reduced, explained_variance = svm_baseline.apply_pca(
        X_train, X_test, n_components=5
    )

    assert X_train_reduced.shape == (30, 5)
    assert X_test_reduced.shape == (10, 5)
    assert 0.0 <= explained_variance <= 1.0


def test_get_label_names():
    assert svm_baseline.get_label_names("4") == list(svm_baseline.config.CLASSES)
    assert len(svm_baseline.get_label_names("12")) == 12


def test_safe_cv_folds_clamps_on_small_classes():
    y = np.array([0, 0, 1, 1, 1, 2])  # class 0 has only 2 members
    assert svm_baseline.safe_cv_folds(y, requested_folds=5) == 2


def test_safe_cv_folds_uses_requested_when_sufficient():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert svm_baseline.safe_cv_folds(y, requested_folds=3) == 3
