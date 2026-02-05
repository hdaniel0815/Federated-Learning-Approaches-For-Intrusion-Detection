"""
Non-IID Data Partitioning for Federated Learning
Two strategies:
1. Realistic: Partition by day/scenario (CIC) or attack type (UNSW)
2. Controlled: Dirichlet distribution with alpha parameter
"""
import pandas as pd
import numpy as np
from scipy.stats import dirichlet
from typing import Dict, List
import json
import os
def partition_cic_by_day(df: pd.DataFrame, num_clients: int = 10) -> Dict:
    """
    Partition CIC-IDS2018 by day (realistic scenario).
    Each day becomes 1-2 clients, creating natural non-IID distribution.
    Returns:
    partition_dict: {client_id: [list of sample indices]}
    """
    print("Partitioning CIC-IDS2018 by day...")
    # Assume we have a 'day' column (add during preprocessing if needed)
    # Or use filename if available in metadata
    # For now, simulate by splitting dataset into 10 temporal chunks
    total_samples = len(df)
    samples_per_client = total_samples // num_clients
    partition = {}
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
        indices = list(range(start_idx, end_idx))
        partition[f"client_{i}"] = indices
        # Print statistics for this client
        client_labels = df.iloc[indices]['label_encoded']
        print(f"\nClient {i}:")
        print(f" Samples: {len(indices):,}")
        print(f" Label distribution:")
        for label, count in client_labels.value_counts().items():
            pct = 100 * count / len(indices)
            print(f" {label}: {count:6,} ({pct:5.2f}%)")
    return partition


def partition_dirichlet(df: pd.DataFrame, num_clients: int, alpha: float) -> Dict:
    """
    Partition using Dirichlet distribution (controlled non-IID).
    Args:
    df: Dataframe with 'label_encoded' column
    num_clients: Number of clients
    alpha: Concentration parameter (lower = more non-IID)
    - 0.1: Highly non-IID
    - 0.5: Moderately non-IID
    - 1.0: Slightly non-IID
    - 10.0: Nearly IID
    Returns:
    partition_dict: {client_id: [list of sample indices]}
    """
    print(f"Partitioning with Dirichlet(alpha={alpha})...")
    num_classes = df['label_encoded'].nunique()
    partition = {f"client_{i}": [] for i in range(num_clients)}
    # For each class, distribute samples to clients using Dirichlet
    for class_id in range(num_classes):
        class_indices = df[df['label_encoded'] == class_id].index.tolist()
        np.random.shuffle(class_indices)
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Convert to split points
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        # Split indices
        splits = np.split(np.array(class_indices), split_points)
        # Assign to clients
        for i, split in enumerate(splits):
            partition[f"client_{i}"].extend(split.tolist())

    # Shuffle each client's data
    for client_id in partition:
        np.random.shuffle(partition[client_id])

    # Print statistics
    print("\nClient statistics:")
    for client_id, indices in partition.items():
        client_labels = df.iloc[indices]['label_encoded']
        print(f"\n{client_id}:")
        print(f" Samples: {len(indices):,}")
        # Calculate entropy (measure of heterogeneity)
        label_counts = client_labels.value_counts(normalize=True)
        entropy = -np.sum(label_counts * np.log(label_counts + 1e-10))
        print(f" Entropy: {entropy:.3f}")
        # Print label distribution
        for label in sorted(client_labels.unique()):
            count = (client_labels == label).sum()
            pct = 100 * count / len(indices)
            print(f" Label {label}: {count:6,} ({pct:5.2f}%)")
    return partition


def save_partition(partition: Dict, output_path: str, metadata: Dict = None):
    """Save partition to JSON file."""
    output = {
        'num_clients': len(partition),
        'total_samples': sum(len(indices) for indices in partition.values()),
        'partition': partition
    }
    if metadata:
        output['metadata'] = metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n Partition saved to: {output_path}")


def main():
    """Main partitioning function."""
    print("="*60)
    print("Data Partitioning for Federated Learning")
    print("="*60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load processed CIC dataset
    print("\nLoading CIC-IDS2018...")
    cic_df = pd.read_parquet('data/cic2018/processed/cic2018_processed.parquet')
    print(f"Loaded {len(cic_df):,} samples")
    
    # Create multiple partitions
    partitions = {
        'cic_realistic': partition_cic_by_day(cic_df, num_clients=10),
        'cic_dirichlet_0.1': partition_dirichlet(cic_df, num_clients=10, alpha=0.1),
        'cic_dirichlet_0.5': partition_dirichlet(cic_df, num_clients=10, alpha=0.5),
        'cic_dirichlet_1.0': partition_dirichlet(cic_df, num_clients=10, alpha=1.0),
    }
    # Save all partitions
    for name, partition in partitions.items():
        output_path = f'partitions/{name}.json'
        metadata = {
            'dataset': 'CIC-IDS2018',
            'strategy': 'realistic' if 'realistic' in name else 'dirichlet',
            'num_clients': 10,
            'seed': 42
        }
        if 'dirichlet' in name:
            alpha = float(name.split('_')[-1])
            metadata['alpha'] = alpha
        save_partition(partition, output_path, metadata)

    print("\n" + "="*60)
    print(" All partitions created successfully!")
    print("="*60)

if __name__ == "__main__":
    main()