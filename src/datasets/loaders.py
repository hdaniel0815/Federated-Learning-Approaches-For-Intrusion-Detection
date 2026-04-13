"""
Each .npz file contains:
    X   — float32 (N, num_features)
    y   — int32   (N,)
    day — unicode (N,)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class ClientNpzDataset(Dataset):
    def __init__(self, parts_dir, parts_to_rows):
        parts_dir = Path(parts_dir)

        part_map = {}
        for item in parts_to_rows:
            part_map[item["part"]] = np.array(item["rows"], dtype=np.int64)

        all_X, all_y = [], []
        for part, rows in part_map.items():
            npz = np.load(parts_dir / part)
            all_X.append(npz["X"][rows])
            all_y.append(npz["y"][rows])

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PublicNpzDataset(Dataset):
    def __init__(self, parts_dir, samples):
        parts_dir = Path(parts_dir)

        from collections import defaultdict
        part_to_rows = defaultdict(list)
        for part_name, row_idx in samples:
            part_to_rows[part_name].append(row_idx)

        all_X, all_y = [], []
        for part_name, rows in part_to_rows.items():
            npz = np.load(parts_dir / part_name)
            all_X.append(npz["X"][rows])
            all_y.append(npz["y"][rows])

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------------
# Loader factories (read from partition manifests — no data leakage)
# ---------------------------------------------------------------------------

def make_test_loader(
    *,
    parts_dir: "str | Path",
    test_manifest_path: "str | Path",
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for the held-out test set.
    Reads the list of test .npz files from test_manifest_path (JSON produced by
    partition_data.py) and exposes every row in those files.
    """
    with open(test_manifest_path) as f:
        manifest = json.load(f)
    parts_dir = Path(parts_dir)
    rows = [
        {"part": entry["part"], "rows": list(range(entry["num_rows"]))}
        for entry in manifest["test_parts"]
    ]
    ds = ClientNpzDataset(parts_dir, rows)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def make_public_loader(
    *,
    parts_dir: "str | Path",
    public_manifest_path: "str | Path",
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for the public (FedMD distillation) set.
    Reads the list of public .npz files from public_manifest_path (JSON produced by
    partition_data.py), guaranteeing no overlap with train or test rows.
    """
    with open(public_manifest_path) as f:
        manifest = json.load(f)
    parts_dir = Path(parts_dir)
    samples: List[Tuple[str, int]] = []
    for entry in manifest["public_parts"]:
        for r in range(entry["num_rows"]):
            samples.append((entry["part"], r))

    ds = PublicNpzDataset(parts_dir, samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
