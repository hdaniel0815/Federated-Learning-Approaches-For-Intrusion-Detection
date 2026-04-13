"""
Shared preprocessing utilities (Stages 2 and 3) — no pyarrow / parquet dependency.

Stage 2: compute global statistics from cleaned .csv.gz files
Stage 3: write final normalised data as compressed NumPy archives (.npz)

Each final .npz file contains:
    X   — float32 array (N, num_features)  — normalised features
    y   — int32  array  (N,)               — encoded class label
    day — unicode array (N,)               — day/file identifier for day-based partitioning
"""
import os, json
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

ATTACK_CAT = "attack_category"
LABEL_ENCODED_COL = "label_encoded"
EXCLUDE_FROM_FEATURES = {ATTACK_CAT, LABEL_ENCODED_COL, "day"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_clean_parts(clean_dir: str) -> List[Path]:
    """Return sorted list of cleaned .csv.gz parts."""
    parts = sorted(Path(clean_dir).glob("*.csv.gz"))
    if not parts:
        raise FileNotFoundError(f"No .csv.gz files found in {clean_dir}")
    return parts


def list_final_parts(final_dir: str) -> List[Path]:
    """Return sorted list of final .npz parts."""
    parts = sorted(Path(final_dir).glob("*.npz"))
    if not parts:
        raise FileNotFoundError(f"No .npz files found in {final_dir}")
    return parts


def safe_std(var: float) -> float:
    return float(np.sqrt(var)) if var > 1e-12 else 1.0


# ---------------------------------------------------------------------------
# Stage 2: compute global statistics and encoders
# ---------------------------------------------------------------------------

def compute_global_stats_and_encoders(clean_csv_dir: str, stats_out_dir: str):
    """
    Single-pass over all cleaned .csv.gz files to compute:
      - per-column mean / std  (numeric_stats.json)
      - categorical value→index maps  (cat_encoders.json)
      - attack-category label→index map  (label_encoder.json)
      - column metadata  (meta.json)
    """
    import pandas as pd

    os.makedirs(stats_out_dir, exist_ok=True)
    parts = list_clean_parts(clean_csv_dir)

    # Read a small sample to discover column types
    first = pd.read_csv(parts[0], nrows=200)
    first.columns = first.columns.str.strip()

    if ATTACK_CAT not in first.columns:
        raise ValueError(f"Expected '{ATTACK_CAT}' column in cleaned dataset. "
                         f"Got: {list(first.columns[:10])}")

    all_cols = [c for c in first.columns if c not in EXCLUDE_FROM_FEATURES]
    num_cols = first[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = first[all_cols].select_dtypes(include=["object", "string"]).columns.tolist()

    sums    = {c: 0.0 for c in num_cols}
    sums_sq = {c: 0.0 for c in num_cols}
    counts  = {c: 0   for c in num_cols}
    cat_values  = {c: set() for c in cat_cols}
    label_values: set = set()

    for part in tqdm(parts, desc="Pass 1: global stats"):
        df = pd.read_csv(part)
        df.columns = df.columns.str.strip()

        label_values.update(df[ATTACK_CAT].dropna().astype(str).unique().tolist())

        if num_cols:
            arr = df[num_cols].to_numpy(dtype=np.float64, copy=False)
            for i, c in enumerate(num_cols):
                col = arr[:, i]
                mask = ~np.isnan(col)
                if mask.any():
                    x = col[mask]
                    sums[c]    += float(x.sum())
                    sums_sq[c] += float((x * x).sum())
                    counts[c]  += int(mask.sum())

        for c in cat_cols:
            if c in df.columns:
                cat_values[c].update(df[c].dropna().astype(str).unique().tolist())

    numeric_stats = {}
    for c in num_cols:
        n = counts[c]
        if n == 0:
            mu, sd = 0.0, 1.0
        else:
            mu  = sums[c] / n
            ex2 = sums_sq[c] / n
            var = max(ex2 - mu * mu, 0.0)
            sd  = safe_std(var)
        numeric_stats[c] = {"mean": float(mu), "std": float(sd)}

    cat_encoders  = {c: {v: i for i, v in enumerate(sorted(vals))}
                     for c, vals in cat_values.items()}
    label_encoder = {v: i for i, v in enumerate(sorted(label_values))}
    feature_cols  = num_cols + cat_cols

    meta = {
        "num_cols":          num_cols,
        "cat_cols":          cat_cols,
        "feature_cols":      feature_cols,
        "num_features":      len(feature_cols),
        "target_col":        ATTACK_CAT,
        "label_encoded_col": LABEL_ENCODED_COL,
        "num_classes":       len(label_encoder),
    }

    with open(os.path.join(stats_out_dir, "numeric_stats.json"), "w") as f:
        json.dump(numeric_stats, f)
    with open(os.path.join(stats_out_dir, "cat_encoders.json"), "w") as f:
        json.dump(cat_encoders, f)
    with open(os.path.join(stats_out_dir, "label_encoder.json"), "w") as f:
        json.dump(label_encoder, f)
    with open(os.path.join(stats_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"\nSaved stats to: {stats_out_dir}")
    print(f"Features: {len(feature_cols)} (numeric={len(num_cols)}, cat={len(cat_cols)}) "
          f"| Classes: {len(label_encoder)}")


# ---------------------------------------------------------------------------
# Stage 3: apply stats → final .npz files
# ---------------------------------------------------------------------------

def transform_to_final_npz(clean_csv_dir: str, final_npz_dir: str, stats_dir: str):
    """
    Z-score normalise numeric cols, integer-encode categorical cols,
    write each part as a compressed NumPy archive (.npz).

    Output arrays per file:
        X   — float32 (N, num_features)
        y   — int32   (N,)
        day — unicode (N,)  — used by day-based partitioning
    """
    import pandas as pd

    os.makedirs(final_npz_dir, exist_ok=True)
    parts = list_clean_parts(clean_csv_dir)

    with open(os.path.join(stats_dir, "numeric_stats.json")) as f:
        numeric_stats = json.load(f)
    with open(os.path.join(stats_dir, "cat_encoders.json")) as f:
        cat_encoders = json.load(f)
    with open(os.path.join(stats_dir, "label_encoder.json")) as f:
        label_encoder = json.load(f)
    with open(os.path.join(stats_dir, "meta.json")) as f:
        meta = json.load(f)

    num_cols     = meta["num_cols"]
    cat_cols     = meta["cat_cols"]
    feature_cols = meta["feature_cols"]

    for idx, part in enumerate(tqdm(parts, desc="Stage 3: transform → .npz")):
        df = pd.read_csv(part)
        df.columns = df.columns.str.strip()

        # Integer-encode categorical features
        for c in cat_cols:
            if c in df.columns:
                mapping = cat_encoders.get(c, {})
                df[c] = df[c].astype(str).map(mapping).fillna(-1).astype(np.float32)

        # Z-score normalise numeric features
        for c in num_cols:
            if c in df.columns:
                mu = float(numeric_stats[c]["mean"])
                sd = float(numeric_stats[c]["std"]) or 1.0
                df[c] = (df[c].astype(np.float32) - mu) / sd

        X   = df[feature_cols].to_numpy(dtype=np.float32)
        y   = (df[ATTACK_CAT].astype(str)
               .map(label_encoder).fillna(-1).astype(np.int32).to_numpy())
        day = (df["day"].astype(str).to_numpy()
               if "day" in df.columns else np.array([""] * len(df), dtype=str))

        out_path = os.path.join(final_npz_dir, f"final_part_{idx:05d}.npz")
        np.savez_compressed(out_path, X=X, y=y, day=day)

    print(f"\nWrote {len(parts)} .npz parts to: {final_npz_dir}")
