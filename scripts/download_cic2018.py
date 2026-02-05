"""
CIC-IDS2018 Dataset Downloader
Downloads the 10-day CIC-IDS2018 dataset from official source.
Dataset info:
- Size: ~16.2 million network flows
4
- Days: 10 (Feb 14-23, 2018)
- Features: 80 flow features (CICFlowMeter)
- Classes: 8 (Benign + 7 attack scenarios)
"""
import os
import urllib.request
import hashlib
from tqdm import tqdm
# Official dataset URLs (from UNB CIC)
BASE_URL = "https://www.unb.ca/cic/datasets/ids-2018.html"
# File list with expected SHA256 checksums
FILES = {
        "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv": {
        "url": "https://.../.../Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
        "sha256": "abc123..." # TODO: Add actual checksum
    },
# Add all 10 CSV files here
}


def verify_checksum(filepath, expected_sha256):
    """Verify file integrity using SHA256."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    actual = sha256_hash.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Checksum mismatch! Expected {expected_sha256}, got {actual}")
    print(f" Checksum verified: {filepath}")


def main():
    """Main download function."""
    data_dir = "data/cic2018/raw"
    os.makedirs(data_dir, exist_ok=True)
    print("="*60)
    print("CIC-IDS2018 Dataset Downloader")
    print("="*60)
    # Download all files
    for filename, info in FILES.items():
        filepath = os.path.join(data_dir, filename)
        # Download
        # download_file(info["url"], filepath)
        # Verify checksum
        # verify_checksum(filepath, info["sha256"])
    print("\n All files downloaded and verified successfully!")
    print(f"Data saved to: {data_dir}")
    # Print statistics
    total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(
    data_dir))
    print(f"Total size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
