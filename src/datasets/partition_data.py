import os, json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

LABEL_COL = "label_encoded"   # must exist in your final parquets

def list_parts(parquet_dir: str) -> List[Path]:
    parts = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in {parquet_dir}")
    return parts

def save_partition(partition: Dict, output_path: str, metadata: Dict = None):
    output = {
        "num_clients": len(partition),
        "total_samples": sum(sum(len(b["rows"]) for b in blocks) for blocks in partition.values()),
        "partition": partition,
    }
    if metadata:
        output["metadata"] = metadata

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved partition to: {output_path}")


def partition_by_day_parquet(parts_dir: str, day_col: str = "day") -> Dict:
    """
    One client per day
    """

    parts = list_parts(parts_dir)

    days = set()
    for p in parts:
        df = pd.read_parquet(p, columns=[day_col])
        days.update(df[day_col].dropna().astype(str).unique())

    days = sorted(days)
    num_clients = len(days)

    print(f"Found {num_clients} days -> {num_clients} clients")

    # indexing days
    day_to_client = {day: i for i, day in enumerate(days)}

    # dictionary of client partitions
    partition = {f"client_{i}": [] for i in range(num_clients)}

    # i should use tqdm here to show progress properly
    for p in parts:
        df = pd.read_parquet(p, columns=[day_col])
        part_name = p.name

        for day, client_idx in day_to_client.items():
            mask = (df[day_col].astype(str) == day)
            if mask.any():
                rows = np.flatnonzero(mask.to_numpy()).tolist()
                if rows:
                    partition[f"client_{client_idx}"].append(
                        {"part": part_name, "rows": rows}
                    )

    return partition

def partition_dirichlet_parquet(parts_dir: str, num_clients: int = 10, alpha: float = 0.5, seed: int = 42) -> Dict:
    """
    Parquet-safe Dirichlet partitioning.
    Returns: {client_id: [{"part": partfile, "rows":[...]} , ...]}
    """
    rng = np.random.default_rng(seed)
    parts = list_parts(parts_dir)

    # PASS 1: count total samples per class
    class_counts = {}
    for p in tqdm(parts, desc="Pass 1: count labels"):
        df = pd.read_parquet(p, columns=[LABEL_COL])
        vc = df[LABEL_COL].value_counts(dropna=False).to_dict()
        for k, c in vc.items():
            # if k is nan, skip
            if pd.isna(k):
                continue
            k = int(k)
            class_counts[k] = class_counts.get(k, 0) + int(c)

    classes = sorted(class_counts.keys())
    if not classes:
        raise ValueError(f"No labels found in column '{LABEL_COL}'.")

    # For each class, sample dirichlet proportions and convert to target counts per client
    targets = {k: None for k in classes}
    for k in classes:
        n = class_counts[k]
        props = rng.dirichlet([alpha] * num_clients)

        # integer targets that sum to n
        t = np.floor(props * n).astype(int)
        remainder = n - int(t.sum())
        if remainder > 0:
            # give leftover to largest fractional parts
            frac = (props * n) - np.floor(props * n)
            for idx in np.argsort(-frac)[:remainder]:
                t[idx] += 1
        targets[k] = t  # shape: (num_clients,)

    # remaining quota per class per client
    remaining = {k: targets[k].copy() for k in classes}

    # output structure: per client, list of blocks {"part":..., "rows":[...]}
    partition = {f"client_{i}": [] for i in range(num_clients)}

    # PASS 2: stream rows and assign to clients while respecting quotas
    for p in tqdm(parts, desc="Pass 2: assign rows"):
        df = pd.read_parquet(p, columns=[LABEL_COL])
        y = df[LABEL_COL].to_numpy()
        part_name = p.name

        # we’ll collect rows per client for this part then append blocks
        rows_for_client = {i: [] for i in range(num_clients)}

        for r, label in enumerate(y):
            if pd.isna(label):
                continue
            k = int(label)
            if k not in remaining:
                continue

            # choose a client that still needs this class
            need = remaining[k]
            candidates = np.flatnonzero(need > 0)

            if candidates.size > 0:
                # pick among candidates (random or greedy)
                ci = int(rng.choice(candidates))
                remaining[k][ci] -= 1
                rows_for_client[ci].append(r)
            else:
                # fallback: everyone already filled quota for this class
                ci = int(rng.integers(0, num_clients))
                rows_for_client[ci].append(r)

        # append part blocks
        for ci, rows in rows_for_client.items():
            if rows:
                partition[f"client_{ci}"].append({"part": part_name, "rows": rows})

    return partition


def main():
    PARTS_DIR = "data/cic2018/processed_final"  # your 79 final parts folder
    OUT_DIR = "partitions"
    os.makedirs(OUT_DIR, exist_ok=True)

    """Main partitioning function."""
    print("="*60)
    print("Data Partitioning for Federated Learning")
    print("="*60)

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Dirichlet partitions
    for a in [0.1, 0.5, 1.0]:
        part = partition_dirichlet_parquet(PARTS_DIR, num_clients=10, alpha=a, seed=42)
        save_partition(
            part,
            os.path.join(OUT_DIR, f"cic_dirichlet_{a}.json"),
            metadata={"dataset":"CIC-IDS2018", "strategy":"dirichlet", "alpha":a, "num_clients":10, "seed":42,
                      "parts_dir": PARTS_DIR, "label_col": LABEL_COL},
        )

    # realistic heterogenuos partitions
    realistic = partition_by_day_parquet(PARTS_DIR, day_col="day")
    save_partition(part, os.path.join(OUT_DIR, "cic_realistic_day.json"),
                  metadata={"dataset":"CIC-IDS2018","strategy":"day","num_clients":10,"seed":42})


    print("\n" + "="*60)
    print(" All partitions created successfully!")
    print("="*60)

if __name__ == "__main__":
    main()