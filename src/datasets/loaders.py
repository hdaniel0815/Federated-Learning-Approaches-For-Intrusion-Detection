"""
Parquet-based Dataset classes and DataLoader factories for federated learning.

These are shared by train_baselines.py and any future training scripts.
All loaders read from partition manifest JSON files produced by partition_data.py,
which guarantees disjoint train / test / public splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class ClientParquetRowsDataset(Dataset):
    """
    Lazy-loading dataset backed by a list of parquet row references.

    parts_to_rows: list of {"part": "final_part_00000.parquet", "rows": [0,1,2,...]}
    Only the specified rows are exposed — no other rows are ever loaded.
    """

    def __init__(
        self,
        parts_dir: "str | Path",
        parts_to_rows: List[dict],
        feature_cols: List[str],
        label_col: str,
    ):
        self.parts_dir    = Path(parts_dir)
        self.feature_cols = list(feature_cols)
        self.label_col    = label_col

        part_map: Dict[str, np.ndarray] = {}
        for item in parts_to_rows:
            part = item["part"]
            rows = np.array(item["rows"], dtype=np.int64)
            part_map[part] = rows
        self.part_to_rows = part_map

        # Flat index: (part_name, position_in_rows_array)
        self.samples: List[Tuple[str, int]] = []
        for part, rows in self.part_to_rows.items():
            for j in range(len(rows)):
                self.samples.append((part, j))
        # Sort by (part, physical row) for sequential parquet reads
        self.samples.sort(key=lambda t: (t[0], int(self.part_to_rows[t[0]][t[1]])))

        self._cache_part: Optional[str] = None
        self._cache_data: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_part(self, part: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_part == part and self._cache_data is not None:
            return self._cache_data
        df = pd.read_parquet(
            self.parts_dir / part,
            columns=self.feature_cols + [self.label_col],
        )
        X = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = df[self.label_col].to_numpy()
        self._cache_part = part
        self._cache_data = (X, y)
        return self._cache_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        part, j = self.samples[idx]
        row_idx = int(self.part_to_rows[part][j])
        X, y    = self._load_part(part)
        return torch.from_numpy(X[row_idx]), torch.tensor(int(y[row_idx]), dtype=torch.long)


class PublicParquetRowsDataset(Dataset):
    """
    Lazy-loading dataset for the public (distillation) set.

    samples: list of (part_name, row_index) tuples built from the public manifest.
    """

    def __init__(
        self,
        parts_dir: "str | Path",
        samples: List[Tuple[str, int]],
        feature_cols: List[str],
        label_col: str,
    ):
        self.parts_dir    = Path(parts_dir)
        self.samples      = samples
        self.feature_cols = list(feature_cols)
        self.label_col    = label_col
        self._cache_part: Optional[str] = None
        self._cache_df: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_part(self, part_name: str) -> pd.DataFrame:
        if self._cache_part == part_name and self._cache_df is not None:
            return self._cache_df
        df = pd.read_parquet(
            self.parts_dir / part_name,
            columns=self.feature_cols + [self.label_col],
        )
        self._cache_part, self._cache_df = part_name, df
        return df

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        part_name, row_idx = self.samples[idx]
        df  = self._load_part(part_name)
        row = df.iloc[int(row_idx)]
        x   = torch.tensor(row[self.feature_cols].to_numpy(dtype=np.float32))
        y   = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Loader factories (read from partition manifests — no data leakage)
# ---------------------------------------------------------------------------

def make_test_loader(
    *,
    parts_dir: "str | Path",
    test_manifest_path: "str | Path",
    feature_cols: List[str],
    label_col: str,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for the held-out test set.
    Reads the list of test parquet files from test_manifest_path (JSON produced by
    partition_data.py) and exposes every row in those files.
    """
    with open(test_manifest_path) as f:
        manifest = json.load(f)
    parts_dir = Path(parts_dir)
    rows = [
        {"part": entry["part"], "rows": list(range(entry["num_rows"]))}
        for entry in manifest["test_parts"]
    ]
    ds = ClientParquetRowsDataset(parts_dir, rows, feature_cols, label_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def make_public_loader(
    *,
    parts_dir: "str | Path",
    public_manifest_path: "str | Path",
    feature_cols: List[str],
    label_col: str,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for the public (FedMD distillation) set.
    Reads the list of public parquet files from public_manifest_path (JSON produced by
    partition_data.py), guaranteeing no overlap with train or test rows.
    """
    with open(public_manifest_path) as f:
        manifest = json.load(f)
    parts_dir = Path(parts_dir)
    samples: List[Tuple[str, int]] = []
    for entry in manifest["public_parts"]:
        for r in range(entry["num_rows"]):
            samples.append((entry["part"], r))

    ds = PublicParquetRowsDataset(parts_dir, samples, feature_cols, label_col)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
