"""
CIC-IDS2018 Dataset Preprocessing
Steps:
1. Load all 10 CSV files
2. Handle missing/inf/nan values
3. Remove identifiers (IP, port, timestamp)
4. Map detailed attack labels to 8 categories
5. Encode categorical features
6. Normalize numerical features
7. Save to parquet format
"""
import pandas as pd
import numpy as np
import os, json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path

# Attack label mapping (detailed categories)
ATTACK_MAPPING = {
    "benign": "Benign",
    "bot": "Bot",
    "ftp-bruteforce": "Brute Force",
    "ssh-bruteforce": "Brute Force",
    "dos attacks-goldeneye": "DoS",
    "dos attacks-hulk": "DoS",
    "dos attacks-slowhttptest": "DoS",
    "dos attacks-slowloris": "DoS",
    "ddos attacks-loic-http": "DDoS",
    "ddos attack-hoic": "DDoS",
    "ddos attack-loic-udp": "DDoS",
    "infilteration": "Infiltration",   # dataset typo
    "brute force -web": "Web Attack",
    "brute force -xss": "Web Attack",
    "sql injection": "Web Attack",
}

# Features to remove (identifiers, redundant, constant)
REMOVE_FEATURES = [
    'Timestamp',
    'Flow ID',
    'Src IP',
    'Dst IP',
    'Src Port',
    'Dst Port',
]

ATTACK_CAT = "attack_category"
LABEL_ENCODED_COL = "label_encoded"
EXCLUDE_FROM_FEATURES = {ATTACK_CAT, LABEL_ENCODED_COL, "day"}

def normalize_label(x: str) -> str:
    return str(x).strip().lower()


def load_and_process_to_parquet_dataset(raw_dir: str, processed_dir: str, chunksize: int = 200_000):
    """
    Stream CSVs in chunks, preprocess each chunk, and write many parquet parts.
    Avoids holding full dataset in memory -> fixes ArrayMemoryError.
    """
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    else:
        print(f"Found {len(csv_files)} CSV files")

    part_idx = 0
    unmapped_seen = set()

    for filename in tqdm(csv_files, desc="Files"):
        filepath = os.path.join(raw_dir, filename)

        # Read in chunks
        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=chunksize):

            chunk.columns = chunk.columns.str.strip()
            # Drop identifiers early (saves memory immediately)
            chunk.drop(columns=[c for c in REMOVE_FEATURES if c in chunk.columns], errors="ignore")

            # Replace inf/-inf -> NaN
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.replace(["Infinity", "inf", "-inf", "NaN", ""], np.nan, inplace=True)

            # Map labels
            if "Label" in chunk.columns:
                label_col = "Label"
                
            elif "label" in chunk.columns:
                label_col = "label"
            else:
                raise ValueError(f"No Label column found in {filename}. Columns: {list(chunk.columns)[:10]}")
            
            raw_norm = chunk[label_col].astype(str).map(normalize_label)

            bad_header_mask = raw_norm.isin({"label"})
            if bad_header_mask.any():
                chunk.loc[~bad_header_mask].copy()
                raw_norm = raw_norm.loc[~bad_header_mask]

            NUMERIC_SENTINAL = "Flow Duration"
            if NUMERIC_SENTINAL in chunk.columns:
                sentinal_numeric = pd.to_numeric(chunk[NUMERIC_SENTINAL], errors="coerce")
                chunk = chunk.loc[sentinal_numeric.notna()].copy()
                raw_norm = raw_norm.loc[chunk.index]
                

            chunk[ATTACK_CAT] = raw_norm.map(ATTACK_MAPPING)

            bad = raw_norm[chunk[ATTACK_CAT].isna()].unique().tolist()
            unmapped_seen.update(bad)

            # Drop unmapped rows
            chunk.dropna(subset=[ATTACK_CAT], inplace=True)

            # Fill NaNs cheaply per chunk (median for numeric, mode for non-numeric)
            # figure this out
            for col in chunk.columns:
                if chunk[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        chunk[col] = chunk[col].fillna(chunk[col].median())
                    else:
                        mode = chunk[col].mode()
                        chunk[col] = chunk[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

            for col in chunk.select_dtypes(include=["float64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="float")
            for col in chunk.select_dtypes(include=["int64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

            chunk["day"] = filename.split("_")[0]
            # Write a parquet “part”
            out_path = os.path.join(processed_dir, f"cleaned_cic2018_part_{part_idx:05d}.parquet")
            chunk.to_parquet(out_path, index=False)  # (compression default ok)
            part_idx += 1

    print(f"Done. Wrote {part_idx} parquet parts to: {processed_dir}")

    unmapped_seen.discard("nan")
    unmapped_seen.discard("none")
    if unmapped_seen:
        print("\nWARNING: Unmapped raw labels were found (these rows were dropped):")
        for x in sorted(unmapped_seen)[:50]:
            print(" -", x)
        print("Consider extending ATTACK_MAPPING if these should be kept.")
    else:
        print("\nALL raw labels were covered by ATTACK_MAPPING")


# global stats stage
def list_parquet_parts(parquet_dir: str) -> List[Path]:
    parts = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    return parts


def safe_std(var: float) -> float:
    return float(np.sqrt(var)) if var > 1e-12 else 1.0


def compute_global_stats_and_encoders(clean_parquet_dir: str, stats_out_dir: str):
    os.makedirs(stats_out_dir, exist_ok=True)
    parts = list_parquet_parts(clean_parquet_dir)

    first = pd.read_parquet(parts[0])
    first.columns = first.columns.str.strip()

    if ATTACK_CAT not in first.columns:
        raise ValueError(f"Expected '{ATTACK_CAT}' column in cleaned dataset")
    
    all_cols = [c for c in first.columns if c not in EXCLUDE_FROM_FEATURES]
    num_cols = first[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = first[all_cols].select_dtypes(include=["object", "string"]).columns.tolist()

    sums = {c: 0.0 for c in num_cols}
    sums_sq = {c: 0.0 for c in num_cols}
    counts = {c: 0 for c in num_cols}
    cat_values = {c: set() for c in cat_cols}
    label_values = set()

    for part in tqdm(parts, desc="Pass 1: Global Stats"):
        df = pd.read_parquet(part)
        df.columns = df.columns.str.strip()

        label_values.update(df[ATTACK_CAT].dropna().astype(str).unique().tolist())

        if num_cols:
            arr = df[num_cols].to_numpy(dtype=np.float64, copy=False)
            for i, c in enumerate(num_cols):
                col = arr[:, i]
                mask = ~np.isnan(col)
                if mask.any():
                    x = col[mask]
                    sums[c] += float(x.sum())
                    sums_sq[c] += float((x*x).sum())
                    counts[c] += int(mask.sum())
    
    for c in cat_cols:
        cat_values[c].update(df[c].dropna().astype(str).unique().tolist())

    numeric_stats = {}
    for c in num_cols:
        n = counts[c]
        if n == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = sums[c] / n
            ex2 = sums_sq[c] / n
            var = max(ex2 - mu * mu, 0.0)
            sd = safe_std(var)
        numeric_stats[c] = {"mean": float(mu), "std": float(sd)}

    cat_encoders = {c: {v: i for i, v in enumerate(sorted(vals))} for c, vals in cat_values.items()}
    label_encoders = {v: i for i, v in enumerate(sorted(label_values))}

    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "target_col": ATTACK_CAT, "label_encoded_col": LABEL_ENCODED_COL}

    with open(os.path.join(stats_out_dir, "numeric_stats.json"), "w") as f:
        json.dump(numeric_stats, f)
    with open(os.path.join(stats_out_dir, "cat_encoders.json"), "w") as f:
        json.dump(cat_encoders, f)
    with open(os.path.join(stats_out_dir, "label_encoder.json"), "w") as f:
        json.dump(label_encoders, f)
    with open(os.path.join(stats_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"\nSaved stats to: {stats_out_dir}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)} | Labels: {len(label_encoders)}")


def transform_to_final_parquet(clean_parquet_dir: str, final_parquet_dir: str, stats_dir:str):
    os.makedirs(final_parquet_dir, exist_ok=True)
    parts = list_parquet_parts(clean_parquet_dir)

    with open(os.path.join(stats_dir, "numeric_stats.json"), "r") as f:
        numeric_stats = json.load(f)
    with open(os.path.join(stats_dir, "cat_encoders.json"), "r") as f:
        cat_encoders = json.load(f)
    with open(os.path.join(stats_dir, "label_encoder.json"), "r") as f:
        label_encoder = json.load(f)
    with open(os.path.join(stats_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    for idx, part in enumerate(tqdm(parts, desc="Stage B2 Pass 2: transform")):
        df = pd.read_parquet(part)
        df.columns = df.columns.str.strip()

        for c in cat_cols:
            mapping = cat_encoders.get(c, {})
            df[c] = df[c].astype(str).map(mapping).fillna(-1).astype(np.int32)

        for c in num_cols:
            mu = float(numeric_stats[c]["mean"])
            sd = float(numeric_stats[c]["std"]) or 1.0
            df[c] = (df[c].astype(np.float32) - mu) / sd

        df[LABEL_ENCODED_COL] = df[ATTACK_CAT].astype(str).map(label_encoder).astype(np.int32)

        out_path = os.path.join(final_parquet_dir, f"final_part_{idx:05d}.parquet")
        df.to_parquet(out_path, index=False, compression="snappy")

    print(f"\nWrote final parquet parts to: {final_parquet_dir}")


# def handle_missing_values(dir_path: str):
#     """Handle missing, inf, and nan values."""
#     print("\nHandling missing values...")


#     # Replace inf with nan
#     print(type(df))
#     df.replace(["Infinity", "inf", "-inf", "NaN", ""], np.nan, inplace=True)
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.fillna(0, inplace=True)

#     # Print missing statistics
#     missing_counts = df.isnull().sum()
#     missing_cols = missing_counts[missing_counts > 0]
#     if len(missing_cols) > 0:
#         print(f"Columns with missing values: {len(missing_cols)}")
#         for col, count in missing_cols.items():
#             pct = 100 * count / len(df)
#             print(f" {col}: {count:,} ({pct:.2f}%)")

#     # Strategy: Drop columns with >50% missing, impute rest with median
#     threshold = 0.5
#     drop_cols = [col for col in df.columns if df[col].isnull().mean() > threshold]
#     if drop_cols:
#         print(f"\nDropping {len(drop_cols)} columns with >{threshold*100}% missing")
#         df = df.drop(columns=drop_cols)

#     # Impute remaining with median (for numerical) or mode (for categorical)
#     for col in df.columns:
#         if df[col].isnull().any():
#             if df[col].dtype in ['float64', 'int64']:
#                 df[col].fillna(df[col].median(), inplace=True)
#             else:
#                 df[col].fillna(df[col].mode()[0], inplace=True)

#     print(f" All missing values handled")
#     return df


# def map_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """Map detailed attack labels to 8 categories."""
#     print("\nMapping attack labels...")

#     # Get label column name (might be 'Label' or 'label')
#     label_col = 'Label' if 'Label' in df.columns else 'label'
     
#     # Print original distribution
#     print("Original label distribution:")
#     print(df[label_col].value_counts())

#     # Map to categories
#     df['attack_category'] = df[label_col].map(ATTACK_MAPPING)

#     # Handle any unmapped labels
#     unmapped = df[df['attack_category'].isnull()]
#     if len(unmapped) > 0:
#         print(f"\nWARNING: {len(unmapped)} unmapped labels found:")
#         print(unmapped[label_col].value_counts())
#         # Set to 'Unknown' or drop
#         df = df.dropna(subset=['attack_category'])

#     # Print final distribution
#     print("\nFinal category distribution:")
#     print(df['attack_category'].value_counts())
#     return df


# def remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
#     """Remove IP addresses, ports, flow IDs, timestamps."""
#     print("\nRemoving identifier columns...")

#     remove_cols = [col for col in REMOVE_FEATURES if col in df.columns]
#     print(f"Removing {len(remove_cols)} columns: {remove_cols}")

#     df = df.drop(columns=remove_cols, errors='ignore')

#     return df


# def encode_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Encode categorical features and normalize numerical."""
#     print("\nEncoding features...")

#     # Separate features and label
#     label_col = 'attack_category'
#     feature_cols = [col for col in df.columns if col != label_col]

#     # Identify categorical columns
#     categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
#     print(f"Categorical columns: {categorical_cols}")

#     # Label encode categorical features
#     encoders = {}
#     for col in categorical_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         encoders[col] = le
        
#     # Normalize numerical features
#     numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
#     print(f"Numerical columns: {len(numerical_cols)}")

#     scaler = StandardScaler()
#     df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#     # Encode target labels
#     label_encoder = LabelEncoder()
#     df['label_encoded'] = label_encoder.fit_transform(df[label_col])

#     print(f"\nClass mapping:")
#     for i, cls in enumerate(label_encoder.classes_):
#         count = (df['label_encoded'] == i).sum()
#         pct = 100 * count / len(df)
#         print(f" {i}: {cls:20s} - {count:10,} ({pct:5.2f}%)")

#     return df, encoders, scaler, label_encoder


# def save_processed(df: pd.DataFrame, output_dir: str):
#     """Save processed data to parquet format."""
#     print(f"\nSaving to {output_dir}...")

#     os.makedirs(output_dir, exist_ok=True)

#     # Save main dataset
#     output_file = os.path.join(output_dir, 'cic2018_processed.parquet')
#     df.to_parquet(output_file, index=False, compression='snappy')

#     print(f" Saved: {output_file}")
#     print(f" Size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
#     print(f" Rows: {len(df):,}")
#     print(f" Columns: {len(df.columns)}")

    
def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("CIC-IDS2018 Preprocessing Pipeline")
    print("="*60)

    raw_dir = "data/cic2018/raw"
    cleaned_dir = "data/cic2018/processed_clean"
    stats_dir = "data/cic2018/stats"
    final_dir = "data/cic2018/processed_final"

    # Step 1: Load all CSVs
    load_and_process_to_parquet_dataset(raw_dir, cleaned_dir, chunksize=200_000)

    # Step 2: Compute Global Stats and Encoders
    compute_global_stats_and_encoders(cleaned_dir, stats_dir)

    # Step 3: Apply Global Stats and Encoders on all parquets
    transform_to_final_parquet(cleaned_dir, final_dir, stats_dir)

    print("\n" + "="*60)
    print(" Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
