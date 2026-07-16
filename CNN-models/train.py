"""CNN training script for four-season colour classification.

Reuses src.dataset.SeasonDataset directly, so training sees exactly the same
crop/illumination-normalize/resize/tensor-normalize pipeline the SVM
baseline was evaluated on -- keeping the comparison in baseline-model/
apples-to-apples.

The train partition (data/processed, partition == "train") is further split
into train/val here (stratified, held constant by --seed) since the dataset
only defines train/test partitions and none is invented upstream. The test
partition is never touched until the final evaluation at the end of main().

Architecture and hyperparameters (model.py, optimizer, lr, batch size,
epochs) are intentionally left simple/placeholder -- fill those in yourself.

Usage:
    python CNN-models/train.py [--epochs N] [--batch-size N] [--lr LR]
                                [--val-frac F] [--limit N] [--num-workers J]
                                [--device D] [--seed S]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> src
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir -> model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Same rationale as baseline-model/svm_baseline.py: DataLoader's default
# "file_descriptor" sharing strategy crashes with a Bus error in this
# sandboxed environment (no /dev/shm). "file_system" avoids it.
torch.multiprocessing.set_sharing_strategy("file_system")

from src import config
from src.dataset import SeasonDataset
from src.plotting import plot_confusion_matrix

from model import SeasonCNN


def get_device(requested=None):
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _season_dataset_labels(dataset):
    """Labels for a SeasonDataset without loading any images."""
    return dataset.df["class"].map(config.CLASS_TO_IDX).to_numpy()


def stratified_limit_indices(dataset, limit, seed=42):
    """A stratified sample of `limit` indices from a SeasonDataset (fast
    smoke test that still spans multiple classes). Returns None if no
    subsampling is needed."""
    n_total = len(dataset)
    if limit is None or limit >= n_total:
        return None
    labels = _season_dataset_labels(dataset)
    indices, _ = train_test_split(
        np.arange(n_total), train_size=limit, stratify=labels, random_state=seed
    )
    return indices


def stratified_train_val_split(dataset, pool_indices, val_frac=0.15, seed=42):
    """Stratified split of `pool_indices` (indices into `dataset`, e.g. a
    --limit subsample or the full range) into train/val index arrays that
    still index directly into `dataset` -- so callers can Subset(dataset,
    train_idx) without stacking a Subset-of-Subset. `dataset` should
    already be the "train" partition; the held-out "test" partition is
    separate and untouched by this function."""
    labels = _season_dataset_labels(dataset)[pool_indices]
    train_idx, val_idx = train_test_split(
        pool_indices, test_size=val_frac, stratify=labels, random_state=seed
    )
    return train_idx, val_idx


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += batch_n

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        n += batch_n
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return total_loss / n, accuracy_score(y_true, y_pred), y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use a stratified subsample of N rows per partition (fast smoke test)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--device", default=None, help="override auto-detected device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    t0 = time.time()
    device = get_device(args.device)

    full_train_ds = SeasonDataset(partition="train")
    full_test_ds = SeasonDataset(partition="test")

    limit_train_idx = stratified_limit_indices(full_train_ds, args.limit, seed=args.seed)
    train_pool_idx = (
        limit_train_idx if limit_train_idx is not None else np.arange(len(full_train_ds))
    )
    limit_test_idx = stratified_limit_indices(full_test_ds, args.limit, seed=args.seed)
    test_pool_idx = (
        limit_test_idx if limit_test_idx is not None else np.arange(len(full_test_ds))
    )

    train_idx, val_idx = stratified_train_val_split(
        full_train_ds, train_pool_idx, val_frac=args.val_frac, seed=args.seed
    )
    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds, val_idx)
    test_ds = Subset(full_test_ds, test_pool_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    label_names = [config.CLASS_DISPLAY_NAMES[c] for c in config.CLASSES]
    model = SeasonCNN(num_classes=len(label_names)).to(device)

    # TODO: these are placeholders -- tune optimizer/lr/loss to taste.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"

    history = []
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"epoch {epoch:>3}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    results = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "device": str(device),
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "accuracy": test_acc,
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(label_names))),
            target_names=label_names, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(label_names)))
        ).tolist(),
        "label_names": label_names,
        "history": history,
        "elapsed_seconds": time.time() - t0,
    }

    out_dir = Path(__file__).resolve().parent
    suffix = "smoketest" if args.limit else None
    out_path = out_dir / (f"results_{suffix}.json" if suffix else "results.json")
    cm_path = out_dir / (f"confusion_matrix_{suffix}.png" if suffix else "confusion_matrix.png")

    plot_confusion_matrix(results["confusion_matrix"], label_names, cm_path)
    results["confusion_matrix_path"] = cm_path.name

    out_path.write_text(json.dumps(results, indent=2))

    print(f"done in {results['elapsed_seconds']:.1f}s; test accuracy = {test_acc:.4f}")
    print(f"wrote {out_path}")
    print(f"wrote {cm_path}")


if __name__ == "__main__":
    main()
