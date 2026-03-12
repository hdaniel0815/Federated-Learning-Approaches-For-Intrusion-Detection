"""
UNSW-NB15 Dataset Preprocessing
Three-stage pipeline:
  Stage 1: Load CSVs → cleaned parquet parts  (dataset-specific)
  Stage 2: Compute global stats & encoders    (shared)
  Stage 3: Z-score normalise → final parquets (shared)

UNSW-NB15 key differences vs CIC datasets:
  - 49 total columns (~43 numeric, 6 categorical: proto, service, state, ...)
  - Label column: 'attack_cat' (not 'Label')
    - NaN in attack_cat means Normal traffic → fill with "Normal" before mapping
  - Binary label column 'label' (0/1) is redundant after attack_cat mapping → dropped
  - Identifier columns: srcip, sport, dstip, dsport
  - No natural 'day' field → use file stem (e.g. UNSW-NB15_1 → "part_1") as proxy
  - 10 attack categories including Normal
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.datasets.preprocess_shared import (
    compute_global_stats_and_encoders,
    transform_to_final_parquet,
    ATTACK_CAT,
)

# ---------------------------------------------------------------------------
# Dataset-specific constants
# ---------------------------------------------------------------------------

ATTACK_MAPPING = {
    "normal":          "Normal",
    "fuzzers":         "Fuzzers",
    "analysis":        "Analysis",
    "backdoors":       "Backdoors",
    "dos":             "DoS",
    "exploits":        "Exploits",
    "generic":         "Generic",
    "reconnaissance":  "Reconnaissance",
    "shellcode":       "Shellcode",
    "worms":           "Worms",
}

# Columns to remove before writing cleaned parquets (identifiers)
REMOVE_FEATURES = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
]

# Binary label col that is redundant after attack_cat mapping
BINARY_LABEL_COL = "label"


def normalize_label(x: str) -> str:
    return str(x).strip().lower()


# ---------------------------------------------------------------------------
# Stage 1: Load CSVs → cleaned parquet parts
# ---------------------------------------------------------------------------

def load_and_process_to_parquet_dataset(raw_dir: str, processed_dir: str, chunksize: int = 200_000):
    """
    Stream UNSW-NB15 CSVs in chunks, preprocess, write cleaned parquet parts.

    UNSW-NB15 provides 4 CSV files (UNSW-NB15_1.csv … UNSW-NB15_4.csv) or
    a pre-split train/test pair. All files are treated uniformly here;
    the partition script handles train/test/public splitting.

    Day proxy: filename stem  (e.g. UNSW-NB15_1.csv → "UNSW-NB15_1")
    """
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    print(f"Found {len(csv_files)} CSV files")

    part_idx = 0
    unmapped_seen: set = set()

    # UNSW-NB15 does not include a header row in some variants; detect by checking
    # whether the first row's 'attack_cat' / 'Attack Cat' column is a string header.
    # We always pass header=0 and handle the edge case below.

    for filename in tqdm(csv_files, desc="Files"):
        filepath = os.path.join(raw_dir, filename)
        day_label = Path(filename).stem   # e.g. "UNSW-NB15_1"

        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=chunksize,
                                  header=0):
            chunk.columns = chunk.columns.str.strip().str.lower()

            # Drop identifier columns early
            chunk.drop(columns=[c for c in REMOVE_FEATURES if c in chunk.columns],
                       inplace=True, errors="ignore")

            # Drop binary label (redundant)
            if BINARY_LABEL_COL in chunk.columns:
                chunk.drop(columns=[BINARY_LABEL_COL], inplace=True)

            # Replace inf / string sentinels with NaN
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.replace(["Infinity", "inf", "-inf", "NaN", ""], np.nan, inplace=True)

            # Locate attack_cat column (UNSW-NB15 uses this name)
            if "attack_cat" in chunk.columns:
                raw_cat = chunk["attack_cat"].astype(str)
            elif "attack cat" in chunk.columns:
                raw_cat = chunk["attack cat"].astype(str)
            else:
                raise ValueError(
                    f"No attack_cat column found in {filename}. "
                    f"Columns: {list(chunk.columns)[:15]}"
                )

            # NaN attack_cat → Normal (many normal rows have blank attack_cat)
            raw_norm = raw_cat.str.strip().str.lower().replace("nan", "normal")

            # Map to canonical category names
            chunk[ATTACK_CAT] = raw_norm.map(ATTACK_MAPPING)
            bad = raw_norm[chunk[ATTACK_CAT].isna()].unique().tolist()
            unmapped_seen.update(bad)
            chunk.dropna(subset=[ATTACK_CAT], inplace=True)

            # Also drop original attack_cat (now superseded by ATTACK_CAT column)
            if "attack_cat" in chunk.columns:
                chunk.drop(columns=["attack_cat"], inplace=True)
            if "attack cat" in chunk.columns:
                chunk.drop(columns=["attack cat"], inplace=True)

            if chunk.empty:
                continue

            # Impute missing values per chunk
            for col in chunk.columns:
                if col == ATTACK_CAT:
                    continue
                if chunk[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        chunk[col] = chunk[col].fillna(chunk[col].median())
                    else:
                        mode = chunk[col].mode()
                        chunk[col] = chunk[col].fillna(
                            mode.iloc[0] if not mode.empty else "Unknown"
                        )

            # Downcast dtypes to save memory
            for col in chunk.select_dtypes(include=["float64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="float")
            for col in chunk.select_dtypes(include=["int64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

            chunk["day"] = day_label

            out_path = os.path.join(
                processed_dir, f"cleaned_unswnb15_part_{part_idx:05d}.parquet"
            )
            chunk.to_parquet(out_path, index=False)
            part_idx += 1

    print(f"Done. Wrote {part_idx} parquet parts to: {processed_dir}")

    unmapped_seen.discard("nan")
    unmapped_seen.discard("none")
    unmapped_seen.discard("")
    if unmapped_seen:
        print("\nWARNING: Unmapped raw labels (rows dropped):")
        for x in sorted(unmapped_seen)[:50]:
            print(" -", x)
        print("Consider extending ATTACK_MAPPING if these should be kept.")
    else:
        print("\nALL raw labels were covered by ATTACK_MAPPING")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("UNSW-NB15 Preprocessing Pipeline")
    print("=" * 60)

    raw_dir     = "data/unswnb15/raw"
    cleaned_dir = "data/unswnb15/processed_clean"
    stats_dir   = "data/unswnb15/stats"
    final_dir   = "data/unswnb15/processed_final"

    # Stage 1
    load_and_process_to_parquet_dataset(raw_dir, cleaned_dir, chunksize=200_000)

    # Stage 2 (shared)
    compute_global_stats_and_encoders(cleaned_dir, stats_dir)

    # Stage 3 (shared)
    transform_to_final_parquet(cleaned_dir, final_dir, stats_dir)

    print("\n" + "=" * 60)
    print(" Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
