"""
CIC-IDS-2018 Dataset Preprocessing
Three-stage pipeline:
  Stage 1: Load CSVs → cleaned .csv.gz parts  (dataset-specific)
  Stage 2: Compute global stats & encoders    (shared)
  Stage 3: Z-score normalise → final .npz     (shared)
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets.preprocess_shared import (
    compute_global_stats_and_encoders,
    transform_to_final_npz,
    ATTACK_CAT,
)

# ---------------------------------------------------------------------------
# Dataset-specific constants
# ---------------------------------------------------------------------------

ATTACK_MAPPING = {
    "benign":                   "Benign",
    "bot":                      "Bot",
    "ftp-bruteforce":           "Brute Force",
    "ssh-bruteforce":           "Brute Force",
    "dos attacks-goldeneye":    "DoS",
    "dos attacks-hulk":         "DoS",
    "dos attacks-slowhttptest": "DoS",
    "dos attacks-slowloris":    "DoS",
    "ddos attacks-loic-http":   "DDoS",
    "ddos attack-hoic":         "DDoS",
    "ddos attack-loic-udp":     "DDoS",
    "infilteration":            "Infiltration",   # dataset typo
    "brute force -web":         "Web Attack",
    "brute force -xss":         "Web Attack",
    "sql injection":            "Web Attack",
}

REMOVE_FEATURES = [
    "Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port",
]


def normalize_label(x: str) -> str:
    return str(x).strip().lower()


# ---------------------------------------------------------------------------
# Stage 1: Load CSVs → cleaned .csv.gz parts
# ---------------------------------------------------------------------------

def load_and_process_to_csv_dataset(raw_dir: str, processed_dir: str, chunksize: int = 200_000):
    """
    Stream CIC-IDS-2018 CSVs in chunks, preprocess each chunk,
    and write cleaned .csv.gz parts (no pyarrow dependency).
    Day label derived from filename stem before first '_'.
    """
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    print(f"Found {len(csv_files)} CSV files")

    part_idx = 0
    unmapped_seen: set = set()

    for filename in tqdm(csv_files, desc="Files"):
        filepath = os.path.join(raw_dir, filename)

        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=chunksize):
            chunk.columns = chunk.columns.str.strip()
            chunk.drop(columns=[c for c in REMOVE_FEATURES if c in chunk.columns],
                       inplace=True, errors="ignore")

            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.replace(["Infinity", "inf", "-inf", "NaN", ""], np.nan, inplace=True)

            if "Label" in chunk.columns:
                label_col = "Label"
            elif "label" in chunk.columns:
                label_col = "label"
            else:
                raise ValueError(
                    f"No Label column found in {filename}. "
                    f"Columns: {list(chunk.columns)[:10]}"
                )

            raw_norm = chunk[label_col].astype(str).map(normalize_label)

            bad_header_mask = raw_norm.isin({"label"})
            if bad_header_mask.any():
                chunk    = chunk.loc[~bad_header_mask].copy()
                raw_norm = raw_norm.loc[chunk.index]

            NUMERIC_SENTINEL = "Flow Duration"
            if NUMERIC_SENTINEL in chunk.columns:
                sentinel_numeric = pd.to_numeric(chunk[NUMERIC_SENTINEL], errors="coerce")
                chunk    = chunk.loc[sentinel_numeric.notna()].copy()
                raw_norm = raw_norm.loc[chunk.index]

            chunk[ATTACK_CAT] = raw_norm.map(ATTACK_MAPPING)
            bad = raw_norm[chunk[ATTACK_CAT].isna()].unique().tolist()
            unmapped_seen.update(bad)
            chunk.dropna(subset=[ATTACK_CAT], inplace=True)

            if chunk.empty:
                continue

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

            for col in chunk.select_dtypes(include=["float64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="float")
            for col in chunk.select_dtypes(include=["int64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

            chunk["day"] = filename.split("_")[0]

            out_path = os.path.join(
                processed_dir, f"cleaned_cic2018_part_{part_idx:05d}.csv.gz"
            )
            chunk.to_csv(out_path, index=False, compression="gzip")
            part_idx += 1

    print(f"Done. Wrote {part_idx} .csv.gz parts to: {processed_dir}")

    unmapped_seen.discard("nan")
    unmapped_seen.discard("none")
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
    print("CIC-IDS-2018 Preprocessing Pipeline")
    print("=" * 60)

    raw_dir     = "data/cic2018/raw"
    cleaned_dir = "data/cic2018/processed_clean"
    stats_dir   = "data/cic2018/stats"
    final_dir   = "data/cic2018/processed_final"

    # Stage 1: raw CSV → cleaned .csv.gz
    load_and_process_to_csv_dataset(raw_dir, cleaned_dir, chunksize=200_000)

    # Stage 2: compute global stats (shared)
    compute_global_stats_and_encoders(cleaned_dir, stats_dir)

    # Stage 3: normalise → final .npz (shared)
    transform_to_final_npz(cleaned_dir, final_dir, stats_dir)

    print("\n" + "=" * 60)
    print(" Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
