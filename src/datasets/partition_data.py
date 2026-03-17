import os, json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

LABEL_COL = "label_encoded"   # must exist in final parquets

# ---------------------------------------------------------------------------
# Per-dataset config
# ---------------------------------------------------------------------------

DATASETS = {
    "cic2018": {
        "parts_dir":  "data/cic2018/processed_final",
        "label_col":  "label_encoded",
        "out_train":  "partitions/cic2018/train",
        "out_test":   "partitions/cic2018/test",
        "out_public": "partitions/cic2018/public",
    },
    "cic2017": {
        "parts_dir":  "data/cic2017/processed_final",
        "label_col":  "label_encoded",
        "out_train":  "partitions/cic2017/train",
        "out_test":   "partitions/cic2017/test",
        "out_public": "partitions/cic2017/public",
    },
    # "unswnb15": {
    #     "parts_dir":  "data/unswnb15/processed_final",
    #     "label_col":  "label_encoded",
    #     "out_train":  "partitions/unswnb15/train",
    #     "out_test":   "partitions/unswnb15/test",
    #     "out_public": "partitions/unswnb15/public",
    # },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_parts(parquet_dir: str) -> List[Path]:
    parts = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in {parquet_dir}")
    return parts


def save_partition(partition: Dict, output_path: str, metadata: Dict = None):
    output = {
        "num_clients":    len(partition),
        "total_samples":  sum(sum(len(b["rows"]) for b in blocks)
                              for blocks in partition.values()),
        "partition":      partition,
    }
    if metadata:
        output["metadata"] = metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved partition to: {output_path}")


def save_test_parts(test_parts: List[Path], output_dir: str, seed: int):
    output = {
        "num_parts":  len(test_parts),
        "test_parts": [
            {"part": p.name, "num_rows": pq.read_metadata(p).num_rows}
            for p in test_parts
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"test_partition_seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved test manifest to: {path}")


def save_public_parts(public_parts: List[Path], output_dir: str, seed: int):
    output = {
        "num_parts":    len(public_parts),
        "public_parts": [
            {"part": p.name, "num_rows": pq.read_metadata(p).num_rows}
            for p in public_parts
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"public_partition_seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved public manifest to: {path}")


# ---------------------------------------------------------------------------
# Three-way disjoint split
# ---------------------------------------------------------------------------

def split_parts_train_test_public(
    parts_dir: str,
    test_frac: float = 0.2,
    public_frac: float = 0.1,
    seed: int = 42,
):
    """
    Returns (train_parts, test_parts, public_parts) — three disjoint Path lists.
    test and public are drawn first so training always gets the majority.
    """
    parts = list_parts(parts_dir)
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(parts))

    n_test   = max(1, round(len(parts) * test_frac))
    n_public = max(1, round(len(parts) * public_frac))

    test_parts   = [parts[i] for i in idx[:n_test]]
    public_parts = [parts[i] for i in idx[n_test : n_test + n_public]]
    train_parts  = [parts[i] for i in idx[n_test + n_public :]]

    print(f"Split: {len(train_parts)} train | {len(test_parts)} test "
          f"| {len(public_parts)} public  (seed={seed})")
    return train_parts, test_parts, public_parts


# ---------------------------------------------------------------------------
# Dirichlet non-IID partitioning
# ---------------------------------------------------------------------------

def partition_dirichlet_parquet(
    parts_dir_or_list: Union[str, List[Path]],
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    label_col: str = LABEL_COL,
) -> Dict:
    """
    Parquet-safe Dirichlet non-IID partitioning.
    Accepts either a directory path string or a pre-filtered list of Path objects
    (the latter allows restricting to train-only files from the three-way split).

    Returns: {client_id: [{"part": filename, "rows": [...]}, ...]}
    """
    rng   = np.random.default_rng(seed)
    parts = (parts_dir_or_list if isinstance(parts_dir_or_list, list)
             else list_parts(parts_dir_or_list))

    # --- Pass 1: count total samples per class ---
    class_counts: Dict[int, int] = {}
    for p in tqdm(parts, desc="Pass 1: count labels"):
        df = pd.read_parquet(p, columns=[label_col])
        vc = df[label_col].value_counts(dropna=False).to_dict()
        for k, c in vc.items():
            if pd.isna(k):
                continue
            k = int(k)
            class_counts[k] = class_counts.get(k, 0) + int(c)

    classes = sorted(class_counts.keys())
    if not classes:
        raise ValueError(f"No labels found in column '{label_col}'.")

    # --- Sample Dirichlet proportions → integer quotas per client per class ---
    targets = {}
    for k in classes:
        n     = class_counts[k]
        props = rng.dirichlet([alpha] * num_clients)
        t     = np.floor(props * n).astype(int)
        remainder = n - int(t.sum())
        if remainder > 0:
            frac = (props * n) - np.floor(props * n)
            for idx in np.argsort(-frac)[:remainder]:
                t[idx] += 1
        targets[k] = t  # shape (num_clients,)

    remaining = {k: targets[k].copy() for k in classes}

    # --- Pass 2: assign rows to clients respecting quotas ---
    partition = {f"client_{i}": [] for i in range(num_clients)}

    for p in tqdm(parts, desc="Pass 2: assign rows"):
        df = pd.read_parquet(p, columns=[label_col])
        y  = df[label_col].to_numpy()
        part_name = p.name

        rows_for_client = {i: [] for i in range(num_clients)}

        for r, label in enumerate(y):
            if pd.isna(label):
                continue
            k = int(label)
            if k not in remaining:
                continue

            need       = remaining[k]
            candidates = np.flatnonzero(need > 0)

            if candidates.size > 0:
                ci = int(rng.choice(candidates))
                remaining[k][ci] -= 1
            else:
                # quota exhausted for all clients; assign randomly (overflow)
                ci = int(rng.integers(0, num_clients))
            rows_for_client[ci].append(r)

        for ci, rows in rows_for_client.items():
            if rows:
                partition[f"client_{ci}"].append({"part": part_name, "rows": rows})

    return partition


# ---------------------------------------------------------------------------
# IID partitioning (baseline)
# ---------------------------------------------------------------------------

def partition_iid_parquet(
    parts_dir_or_list: Union[str, List[Path]],
    num_clients: int = 10,
    seed: int = 42,
    label_col: str = LABEL_COL,
) -> Dict:
    """
    Uniform random (IID) split across clients.
    Used as the IID baseline to confirm non-IID is the cause of degradation.
    """
    rng   = np.random.default_rng(seed)
    parts = (parts_dir_or_list if isinstance(parts_dir_or_list, list)
             else list_parts(parts_dir_or_list))

    # Collect (part_name, row_index) pairs and shuffle globally
    all_rows: List[tuple] = []
    for p in tqdm(parts, desc="IID: collecting rows"):
        df = pd.read_parquet(p, columns=[label_col])
        for r in range(len(df)):
            all_rows.append((p.name, r))

    perm = rng.permutation(len(all_rows))
    splits = np.array_split(perm, num_clients)

    partition = {f"client_{i}": [] for i in range(num_clients)}
    for ci, split_idx in enumerate(splits):
        # Group contiguous rows per part to avoid huge row lists
        from collections import defaultdict
        part_to_rows: dict = defaultdict(list)
        for idx in split_idx:
            part_name, row = all_rows[idx]
            part_to_rows[part_name].append(row)
        for part_name, rows in part_to_rows.items():
            partition[f"client_{ci}"].append({"part": part_name, "rows": sorted(rows)})

    return partition


# ---------------------------------------------------------------------------
# Realistic day-based partitioning
# ---------------------------------------------------------------------------

def partition_by_day_parquet(
    parts_dir_or_list: Union[str, List[Path]],
    day_col: str = "day",
) -> Dict:
    """One client per unique day value (natural heterogeneity)."""
    parts = (parts_dir_or_list if isinstance(parts_dir_or_list, list)
             else list_parts(parts_dir_or_list))

    days: set = set()
    for p in parts:
        df = pd.read_parquet(p, columns=[day_col])
        days.update(df[day_col].dropna().astype(str).unique())

    days = sorted(days)
    day_to_client = {day: i for i, day in enumerate(days)}
    num_clients   = len(days)
    print(f"Found {num_clients} unique days → {num_clients} clients")

    partition = {f"client_{i}": [] for i in range(num_clients)}

    for p in tqdm(parts, desc="Day partition"):
        df = pd.read_parquet(p, columns=[day_col])
        part_name = p.name
        for day, ci in day_to_client.items():
            mask = df[day_col].astype(str) == day
            if mask.any():
                rows = np.flatnonzero(mask.to_numpy()).tolist()
                if rows:
                    partition[f"client_{ci}"].append({"part": part_name, "rows": rows})

    return partition


# ---------------------------------------------------------------------------
# Main: generate all partitions for all datasets
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Data Partitioning for Federated Learning")
    print("=" * 60)

    seeds  = [42, 123, 456]
    alphas = [0.1, 0.5, 1.0]

    for ds_name, cfg in DATASETS.items():
        parts_dir  = cfg["parts_dir"]
        label_col  = cfg["label_col"]
        out_train  = cfg["out_train"]
        out_test   = cfg["out_test"]
        out_public = cfg["out_public"]

        # Check dataset exists
        if not list(Path(parts_dir).glob("*.parquet")):
            print(f"\n[SKIP] {ds_name}: no parquet files in {parts_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        for seed in seeds:
            train_parts, test_parts, public_parts = split_parts_train_test_public(
                parts_dir, test_frac=0.2, public_frac=0.1, seed=seed
            )
            save_test_parts(test_parts, out_test, seed)
            save_public_parts(public_parts, out_public, seed)

            # Dirichlet non-IID partitions on train parts only
            for alpha in alphas:
                part = partition_dirichlet_parquet(
                    train_parts,
                    num_clients=10,
                    alpha=alpha,
                    seed=seed,
                    label_col=label_col,
                )
                save_partition(
                    part,
                    os.path.join(out_train, f"{ds_name}_dirichlet_{alpha}_seed_{seed}.json"),
                    metadata={
                        "dataset": ds_name, "strategy": "dirichlet",
                        "alpha": alpha, "num_clients": 10, "seed": seed,
                        "parts_dir": parts_dir, "label_col": label_col,
                    },
                )

            # IID baseline partition
            iid_part = partition_iid_parquet(
                train_parts, num_clients=10, seed=seed, label_col=label_col
            )
            save_partition(
                iid_part,
                os.path.join(out_train, f"{ds_name}_iid_seed_{seed}.json"),
                metadata={
                    "dataset": ds_name, "strategy": "iid",
                    "num_clients": 10, "seed": seed,
                    "parts_dir": parts_dir, "label_col": label_col,
                },
            )

        # Realistic day-based partition (seed-independent; uses all train parts from seed=42)
        train_parts_42, _, _ = split_parts_train_test_public(
            parts_dir, test_frac=0.2, public_frac=0.1, seed=42
        )
        realistic = partition_by_day_parquet(train_parts_42, day_col="day")
        save_partition(
            realistic,
            os.path.join(out_train, f"{ds_name}_realistic_day.json"),
            metadata={
                "dataset": ds_name, "strategy": "day",
                "num_clients": len(realistic), "seed": 42,
                "parts_dir": parts_dir,
            },
        )

    print("\n" + "=" * 60)
    print(" All partitions created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
