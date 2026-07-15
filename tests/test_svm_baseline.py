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


class _TinySmallImageDataset(torch.utils.data.Dataset):
    """Yields 64x64 images, unlike _TinyDataset's 224x224 -- exercises that
    dataset_to_design_matrix infers feature dim from the data rather than a
    hardcoded constant, since main() now varies resolution via --img-size."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.full((3, 64, 64), float(idx), dtype=torch.float32)
        return image, torch.tensor(idx % 4, dtype=torch.long)


def test_dataset_to_design_matrix_infers_feature_dim_from_data():
    X, y = svm_baseline.dataset_to_design_matrix(_TinySmallImageDataset(5))
    assert X.shape == (5, 3 * 64 * 64)


class _FakeSeasonDataset:
    """Mimics the .df surface stratified_limit_indices needs, without
    depending on real data/processed/ files."""

    def __init__(self, classes):
        self.df = pd.DataFrame({"class": classes})

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


def test_safe_cv_folds_clamps_on_small_classes():
    y = np.array([0, 0, 1, 1, 1, 2])  # class 0 has only 2 members
    assert svm_baseline.safe_cv_folds(y, requested_folds=5) == 2


def test_safe_cv_folds_uses_requested_when_sufficient():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert svm_baseline.safe_cv_folds(y, requested_folds=3) == 3
