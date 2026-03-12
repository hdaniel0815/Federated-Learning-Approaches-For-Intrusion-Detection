"""
Federated Learning baseline training script.

Runs FedAvg, FedProx, FedMD, and FedProtoKD on each dataset × partition × model
combination, collecting Dan-H-style parallel-list histories with bandwidth tracking.

Execution order:
  1. python src/datasets/preprocess_cic2017.py   (or cic / unswnb15)
  2. python src/datasets/partition_data.py
  3. python scripts/train_baselines.py
"""
from __future__ import annotations

import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")     # non-interactive backend (safe on Windows / headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.fl.base import ServerPayload, ClientUpdate
from src.fl.fedavg import FedAvgAlgorithm, FedAvgPlugin
from src.fl.fedprox import FedProxAlgorithm, FedProxPlugin
from src.fl.fedmd import FedMDAlgorithm, FedMDPlugin
from src.fl.fedprotokd import FedProtoKDAlgorithm, FedProtoKDPlugin

from src.models.mlp import StudentMLP
from src.models.lstm import StudentLSTM
from src.models.cnn1d import StudentCNN1D
from src.models.transformer import StudentTabTransformer

import src.eval.metrics

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]

ACCURACY_THRESHOLD = 0.90   # for rounds_to_threshold metric

MODEL_BUILDERS = {
    "lstm":        lambda input_dim, num_classes: StudentLSTM(input_dim, num_classes),
    "transformer": lambda input_dim, num_classes: StudentTabTransformer(input_dim, num_classes),
}

# Per-dataset paths (all three datasets supported)
DATASETS: Dict[str, Dict[str, str]] = {
    "cic2018": {
        "stats_dir":    "data/cic2018/stats",
        "parquets":     "data/cic2018/processed_final",
        "train_parts":  "partitions/cic2018/train",
        "test_parts":   "partitions/cic2018/test",
        "public_parts": "partitions/cic2018/public",
    },
    "cic2017": {
        "stats_dir":    "data/cic2017/stats",
        "parquets":     "data/cic2017/processed_final",
        "train_parts":  "partitions/cic2017/train",
        "test_parts":   "partitions/cic2017/test",
        "public_parts": "partitions/cic2017/public",
    },
    "unswnb15": {
        "stats_dir":    "data/unswnb15/stats",
        "parquets":     "data/unswnb15/processed_final",
        "train_parts":  "partitions/unswnb15/train",
        "test_parts":   "partitions/unswnb15/test",
        "public_parts": "partitions/unswnb15/public",
    },
}

# Strategy color palette (matches Dan-H-codes-v1 convention)
STRATEGY_COLORS = {
    "fedavg":     "blue",
    "fedprox":    "green",
    "fedmd":      "orange",
    "fedprotokd": "red",
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_std_str(values: List[float]) -> str:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return "N/A"
    if len(x) == 1:
        return f"{float(x[0]):.4f} ± 0.0000"
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"


def log(stage: str, **kv: Any) -> None:
    msg = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{stage}] {msg}")


def _safe_stem(s: str) -> str:
    """Replace characters unsafe for filenames."""
    return re.sub(r"[^\w\-]", "_", s)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ClientParquetRowsDataset(Dataset):
    """
    parts_to_rows: list of {"part": "final_part_00000.parquet", "rows": [1,2,3,...]}
    """
    def __init__(
        self,
        parts_dir: Path,
        parts_to_rows: List[dict],
        feature_cols: List[str],
        label_col: str,
    ):
        self.parts_dir   = Path(parts_dir)
        self.feature_cols = list(feature_cols)
        self.label_col   = label_col

        part_map: Dict[str, np.ndarray] = {}
        for item in parts_to_rows:
            part = item["part"]
            rows = np.array(item["rows"], dtype=np.int64)
            part_map[part] = rows
        self.part_to_rows = part_map

        self.samples: List[Tuple[str, int]] = []
        for part, rows in self.part_to_rows.items():
            for j in range(len(rows)):
                self.samples.append((part, j))
        self.samples.sort(key=lambda t: (t[0], int(self.part_to_rows[t[0]][t[1]])))

        self._cache_part: Optional[str] = None
        self._cache_df: Optional[Tuple] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_part(self, part: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_part == part and self._cache_df is not None:
            return self._cache_df
        df = pd.read_parquet(
            self.parts_dir / part,
            columns=self.feature_cols + [self.label_col],
        )
        X = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = df[self.label_col].to_numpy()
        self._cache_part = part
        self._cache_df   = (X, y)
        return self._cache_df

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        part, j  = self.samples[idx]
        row_idx  = int(self.part_to_rows[part][j])
        X, y     = self._load_part(part)
        return torch.from_numpy(X[row_idx]), torch.tensor(int(y[row_idx]), dtype=torch.long)


class PublicParquetRowsDataset(Dataset):
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
        self._cache_part  = None
        self._cache_df    = None

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
# Loader factories
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
    """Build a DataLoader for the held-out test set from the test manifest JSON."""
    with open(test_manifest_path) as f:
        meta = json.load(f)
    parts_dir = Path(parts_dir)
    rows = [
        {
            "part": p,
            "rows": list(range(pq.ParquetFile(parts_dir / p).metadata.num_rows)),
        }
        for p in meta["test_parts"]
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
    Build a DataLoader for the public (distillation) set from the public manifest JSON.
    Using a manifest guarantees no overlap with train or test sets.
    """
    with open(public_manifest_path) as f:
        meta = json.load(f)
    parts_dir = Path(parts_dir)
    samples: List[Tuple[str, int]] = []
    for part_name in meta["public_parts"]:
        n = pq.ParquetFile(parts_dir / part_name).metadata.num_rows
        for r in range(n):
            samples.append((part_name, r))

    ds = PublicParquetRowsDataset(parts_dir, samples, feature_cols, label_col)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# FL Client
# ---------------------------------------------------------------------------

class FLClient:
    def __init__(
        self,
        cid: int,
        model: torch.nn.Module,
        train_loader: DataLoader,
        *,
        device: Optional[str] = None,
        public_loader: Optional[DataLoader] = None,
        distill_lambda: float = 1.0,
        distill_T: float = 2.0,
        mu: float = 0.0,
    ):
        self.cid = cid
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model         = model.to(self.device)
        self.train_loader  = train_loader
        self.public_loader = public_loader
        self.distill_lambda = distill_lambda
        self.distill_T     = distill_T
        self.mu            = mu
        self.plugins: List[Any] = []

    def set_plugins(self, plugins: List[Any]) -> None:
        self.plugins = plugins

    def local_train(
        self,
        payload: ServerPayload,
        *,
        local_epochs: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> ClientUpdate:
        self._apply_payload(payload)

        opt     = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        ce      = torch.nn.CrossEntropyLoss()
        plugins = self.plugins

        for p in plugins:
            p.on_round_start(self, payload)

        total_seen = 0
        self.model.train()

        for _ in tqdm(range(local_epochs), desc="local training progress"):
            for x, y in self.train_loader:
                x, y  = x.to(self.device), y.to(self.device)
                batch = (x, y)

                opt.zero_grad()
                logits, protos, feats = self._forward_pack(x)

                loss = ce(logits, y)
                for p in plugins:
                    loss = loss + p.extra_loss(self, batch, (logits, protos, feats), payload)

                loss.backward()
                opt.step()

                for p in plugins:
                    p.after_step(self, batch, (logits, protos, feats), payload)

                total_seen += x.size(0)

        extra: Dict[str, Any] = {}
        for p in plugins:
            extra.update(p.on_round_end(self, payload))

        weights    = self._get_weights_cpu()
        pub_logits = extra.get("pub_logits")
        prototypes = extra.get("prototypes")

        # Bandwidth: count only what the server actually receives
        if pub_logits is not None:
            # FedMD: server receives logits, not weights
            bytes_sent = int(pub_logits.nbytes)
        else:
            bytes_sent = sum(v.nelement() * v.element_size() for v in weights.values())
            if prototypes is not None:
                bytes_sent += sum(
                    np.array(v, dtype=np.float32).nbytes for v in prototypes.values()
                )

        return ClientUpdate(
            client_id=self.cid,
            n_samples=total_seen,
            weights=weights,
            pub_logits=pub_logits,
            prototypes=prototypes,
            bytes_sent=bytes_sent,
        )

    def _apply_payload(self, payload: ServerPayload) -> None:
        if payload.weights is not None:
            self.model.load_state_dict(payload.weights, strict=True)

    def _forward_pack(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            out = self.model(x, return_all=True)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                return out[0], out[1], out[2]
        except TypeError:
            pass
        logits = self.model(x)
        return logits, logits, logits

    def _get_weights_cpu(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}


# ---------------------------------------------------------------------------
# Manifest / dataset helpers
# ---------------------------------------------------------------------------

def load_partition_manifest(path: Path) -> Tuple[Dict[int, List[dict]], int]:
    with open(path) as f:
        manifest = json.load(f)
    partitions  = manifest["partition"]
    num_clients = int(manifest["num_clients"])
    client_map  = {
        int(cid.split("_")[1]): part_list
        for cid, part_list in partitions.items()
    }
    return client_map, num_clients


def build_client_datasets(
    client_map: Dict[int, List[dict]],
    *,
    parts_dir: Path,
    feature_cols: List[str],
    label_col: str,
) -> Dict[int, Dataset]:
    return {
        cid: ClientParquetRowsDataset(parts_dir, part_list, feature_cols, label_col)
        for cid, part_list in client_map.items()
    }


# ---------------------------------------------------------------------------
# Federated learning loop
# ---------------------------------------------------------------------------

def run_federated_learning(
    *,
    strategy,
    clients: List[FLClient],
    init_server_payload: ServerPayload,
    num_rounds: int,
    clients_per_round: int,
    eval_fn=None,
    seed: int = 42,
) -> Tuple[ServerPayload, dict, dict]:
    rng        = random.Random(seed)
    all_cids   = list(range(len(clients)))
    global_state = init_server_payload

    history: dict = {
        "rounds": [], "accuracy": [], "f1_macro": [],
        "f1_weighted": [], "bytes_communicated": [],
    }
    cumulative_bytes = 0
    last_eval_m: dict = {}

    pbar = tqdm(range(1, num_rounds + 1), desc=strategy.__class__.__name__)
    for rnd in pbar:
        k        = min(clients_per_round, len(all_cids))
        selected = rng.sample(all_cids, k)
        payload  = strategy.server_payload(global_state)

        updates: List[ClientUpdate] = []
        for cid in selected:
            updates.append(clients[cid].local_train(payload))

        global_state = strategy.aggregate(global_state, updates)

        # Accumulate bandwidth
        round_bytes      = sum(u.bytes_sent for u in updates)
        cumulative_bytes += round_bytes

        # Evaluate on round 1, every 5 rounds, and the final round
        if eval_fn is not None and (rnd == 1 or rnd % 5 == 0 or rnd == num_rounds):
            m = eval_fn(global_state, clients)
            if m:
                last_eval_m = m
                history["rounds"].append(rnd)
                history["accuracy"].append(m.get("accuracy", 0.0))
                history["f1_macro"].append(m.get("f1_macro", 0.0))
                history["f1_weighted"].append(m.get("f1_weighted", 0.0))
                history["bytes_communicated"].append(cumulative_bytes)
                pbar.set_postfix(
                    {k: f"{v:.4f}" for k, v in m.items() if isinstance(v, float)}
                )

    final_results: dict = {}
    if history["accuracy"]:
        final_results = {
            "accuracy":            history["accuracy"][-1],
            "f1_macro":            history["f1_macro"][-1],
            "f1_weighted":         history["f1_weighted"][-1],
            "rounds_to_threshold": next(
                (r for r, a in zip(history["rounds"], history["accuracy"])
                 if a >= ACCURACY_THRESHOLD),
                None,
            ),
            "total_bytes":         cumulative_bytes,
            "y_true":              last_eval_m.get("y_true", []),
            "y_pred":              last_eval_m.get("y_pred", []),
        }

    return global_state, history, final_results


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def train_baselines(
    *,
    strategies_to_run: List[str],
    clients: List[FLClient],
    num_rounds: int,
    clients_per_round: int,
    num_classes: int,
    public_loader: Optional[DataLoader] = None,
    seed: int = 42,
    eval_fn=None,
) -> Dict[str, dict]:
    init_client_weights = [copy.deepcopy(c.model.state_dict()) for c in clients]
    template_weights    = copy.deepcopy(init_client_weights[0])

    def init_payload(name: str) -> ServerPayload:
        if name in ("fedavg", "fedprox", "fedprotokd"):
            return ServerPayload(weights=copy.deepcopy(template_weights))
        if name == "fedmd":
            if public_loader is None:
                raise ValueError("FedMD requires public_loader.")
            n_pub = len(public_loader.dataset)
            return ServerPayload(
                pub_logits=np.zeros((n_pub, num_classes), dtype=np.float32)
            )
        raise ValueError(f"Unknown strategy: {name}")

    strategy_builders = {
        "fedavg":     lambda: FedAvgAlgorithm(initial_weights=copy.deepcopy(template_weights)),
        "fedprox":    lambda: FedProxAlgorithm(initial_weights=copy.deepcopy(template_weights), mu=0.01),
        "fedmd":      lambda: FedMDAlgorithm(
            n_pub=(len(public_loader.dataset) if public_loader else 0),
            num_classes=num_classes,
        ),
        "fedprotokd": lambda: FedProtoKDAlgorithm(initial_weights=copy.deepcopy(template_weights)),
    }

    results: Dict[str, dict] = {}

    # I can probably rework this with composition
    for name in strategies_to_run:
        # Reset all clients to same initial weights
        for i, c in enumerate(clients):
            c.model.load_state_dict(copy.deepcopy(init_client_weights[i]), strict=True)

        if name == "fedavg":
            plugins = [FedAvgPlugin()]
        elif name == "fedprox":
            plugins = [FedAvgPlugin(), FedProxPlugin(mu=0.01)]
        elif name == "fedmd":
            plugins = [FedAvgPlugin(), FedMDPlugin(T=2.0, kd_lambda=1.0)]
        elif name == "fedprotokd":
            plugins = [FedAvgPlugin(), FedProtoKDPlugin(alpha_proto=0.3)]
        else:
            plugins = []

        for c in clients:
            c.set_plugins(plugins)

        strategy           = strategy_builders[name]()
        init_server_payload = init_payload(name)

        server_state, history, final_results = run_federated_learning(
            strategy=strategy,
            clients=clients,
            init_server_payload=init_server_payload,
            num_rounds=num_rounds,
            clients_per_round=clients_per_round,
            eval_fn=eval_fn,
            seed=seed,
        )
        results[name] = {
            "server_state":  server_state,
            "history":       history,
            "final_results": final_results,
        }

    return results


# ---------------------------------------------------------------------------
# Eval function factory
# ---------------------------------------------------------------------------

def make_eval_fn(model_arch, input_dim: int, num_classes: int, test_loader, device: str):
    """
    Returns an eval_fn(server_state, clients) -> dict.
    For FedMD (server_state.weights is None), averages client weights for evaluation.
    Returns accuracy, f1_macro, f1_weighted, y_true (list), y_pred (list).
    """
    def eval_fn(server_state: ServerPayload, clients: List[FLClient]) -> dict:
        if server_state.weights is not None:
            weights = server_state.weights
        else:
            # FedMD has no global model — use weighted average of client models
            total = sum(len(c.train_loader.dataset) for c in clients)
            if total == 0:
                return {}
            w_avg = None
            for c in clients:
                frac = len(c.train_loader.dataset) / total
                w    = {k: v.detach().cpu().float() for k, v in c.model.state_dict().items()}
                if w_avg is None:
                    w_avg = {k: w[k] * frac for k in w}
                else:
                    for k in w:
                        w_avg[k] = w_avg[k] + w[k] * frac
            weights = w_avg

        if weights is None:
            return {}

        tmp = model_arch(input_dim=input_dim, num_classes=num_classes)
        tmp.load_state_dict(weights, strict=False)

        y_true, y_pred = src.eval.metrics.infer(tmp, test_loader, device)
        m              = src.eval.metrics.compute_metrics(y_true, y_pred)
        return {
            "accuracy":   m["accuracy"],
            "f1_macro":   m["f1_macro"],
            "f1_weighted":m["f1_weighted"],
            "y_true":     y_true.tolist(),
            "y_pred":     y_pred.tolist(),
        }

    return eval_fn


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curves(histories: Dict[str, dict], save_path: str) -> None:
    """
    Three-panel figure: Accuracy | F1-macro | Cumulative bytes vs rounds.
    histories: {strategy_name: history_dict}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("accuracy",           "Accuracy",                "Accuracy"),
        ("f1_macro",           "F1-Macro",                "F1-Macro"),
        ("bytes_communicated", "Cumulative Bytes (MB)",   "Bandwidth"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for strat, hist in histories.items():
            rounds = hist.get("rounds", [])
            values = hist.get(key, [])
            if not rounds or not values:
                continue
            if key == "bytes_communicated":
                values = [v / 1e6 for v in values]   # bytes → MB
            ax.plot(
                rounds, values,
                label=strat,
                color=STRATEGY_COLORS.get(strat, None),
                marker="o", markersize=3,
            )
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_comparison_table(
    table_data: Dict[str, dict],
    save_path: str,
) -> None:
    """
    Print and save a comparison table.
    table_data: {strategy: {"accuracy", "f1_macro", "f1_weighted",
                             "rounds_to_threshold" (list), "total_bytes"}}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'Method':<14} | {'Accuracy':>10} | {'F1-Macro':>10} "
        f"| {'F1-Weighted':>12} | {'Rounds-to-90%':>14} | {'Total-Bytes (MB)':>17}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for strat, vals in table_data.items():
        rtt = vals.get("rounds_to_threshold", [None])
        if isinstance(rtt, list):
            non_null = [r for r in rtt if r is not None]
            rtt_str  = str(round(np.mean(non_null))) if non_null else "N/A"
        else:
            rtt_str = str(rtt) if rtt is not None else "N/A"

        tb_mb = vals.get("total_bytes", 0) / 1e6

        lines.append(
            f"{strat:<14} | {vals['accuracy']:>10.4f} | {vals['f1_macro']:>10.4f} "
            f"| {vals['f1_weighted']:>12.4f} | {rtt_str:>14} | {tb_mb:>17.2f}"
        )

    lines.append(sep)
    table_str = "\n".join(lines)

    print(table_str)
    with open(save_path, "w") as f:
        f.write(table_str + "\n")


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    HP = {
        "num_rounds":       5,
        "clients_per_round": 10,
        "batch_size":       1024,
        "lr":               1e-3,
    }

    strategies_to_run_base = ["fedprotokd"]

    for ds_name, ds_cfg in DATASETS.items():
        parquets_dir = Path(ds_cfg["parquets"])

        # Skip if the dataset hasn't been preprocessed yet
        if not list(parquets_dir.glob("*.parquet")):
            print(f"\n[SKIP] {ds_name}: no parquet files in {parquets_dir}")
            continue

        # Load dataset metadata
        stats_dir = ds_cfg["stats_dir"]
        with open(f"{stats_dir}/meta.json") as f:
            meta = json.load(f)
        with open(f"{stats_dir}/label_encoder.json") as f:
            label_encoder = json.load(f)

        feature_cols = meta["num_cols"] + meta["cat_cols"]
        label_col    = meta["label_encoded_col"]
        num_classes  = len(label_encoder)
        class_names  = [k for k, _ in sorted(label_encoder.items(), key=lambda x: x[1])]

        log("DATASET", name=ds_name, features=len(feature_cols), classes=num_classes)

        # Discover partition files and group by type (strip _seed_N suffix)
        train_dir       = Path(ds_cfg["train_parts"])
        partition_files = sorted(train_dir.glob("*.json"))
        if not partition_files:
            print(f"[SKIP] {ds_name}: no partition JSON files in {train_dir}")
            continue

        type_to_files: Dict[str, List[Path]] = defaultdict(list)
        for pf in partition_files:
            ptype = re.sub(r"_seed_\d+$", "", pf.stem)
            type_to_files[ptype].append(pf)

        for model_name, model_arch in MODEL_BUILDERS.items():
            log("MODEL_START", dataset=ds_name, model=model_name)

            for ptype, pfiles in sorted(type_to_files.items()):
                log("PARTITION_TYPE", dataset=ds_name, model=model_name, partition=ptype)

                type_seed_results: Dict[int, Dict[str, dict]] = {}

                for pf in sorted(pfiles):
                    seed_m = re.search(r"_seed_(\d+)$", pf.stem)
                    seed   = int(seed_m.group(1)) if seed_m else 42

                    set_all_seeds(seed)
                    log("SEED_START", partition=pf.stem, seed=seed)

                    # --- Test loader (disjoint from train + public) ---
                    test_manifest = Path(ds_cfg["test_parts"]) / f"test_partition_seed_{seed}.json"
                    if not test_manifest.exists():
                        print(f"  [WARN] Missing test manifest: {test_manifest} — skipping seed {seed}")
                        continue
                    test_loader = make_test_loader(
                        parts_dir=parquets_dir,
                        test_manifest_path=test_manifest,
                        feature_cols=feature_cols,
                        label_col=label_col,
                        batch_size=HP["batch_size"],
                    )

                    # --- Public loader (disjoint; required for FedMD) ---
                    public_manifest = (
                        Path(ds_cfg["public_parts"]) / f"public_partition_seed_{seed}.json"
                    )
                    public_loader: Optional[DataLoader] = None
                    if public_manifest.exists():
                        public_loader = make_public_loader(
                            parts_dir=parquets_dir,
                            public_manifest_path=public_manifest,
                            feature_cols=feature_cols,
                            label_col=label_col,
                            batch_size=256,
                        )

                    # Decide which strategies to run this seed
                    strats = list(strategies_to_run_base)
                    if public_loader is not None:
                        strats.append("fedmd")

                    # --- Client data ---
                    client_map, num_clients = load_partition_manifest(pf)
                    client_datasets = build_client_datasets(
                        client_map,
                        parts_dir=parquets_dir,
                        feature_cols=feature_cols,
                        label_col=label_col,
                    )
                    client_loaders = {
                        cid: DataLoader(
                            ds,
                            batch_size=HP["batch_size"],
                            shuffle=False,
                            num_workers=0,
                            pin_memory=torch.cuda.is_available(),
                        )
                        for cid, ds in client_datasets.items()
                    }

                    # --- Build FL clients ---
                    fl_clients: List[FLClient] = [
                        FLClient(
                            cid=cid,
                            model=model_arch(
                                input_dim=len(feature_cols), num_classes=num_classes
                            ),
                            train_loader=client_loaders[cid],
                            device=device,
                            public_loader=public_loader,
                        )
                        for cid in range(num_clients)
                    ]

                    eval_fn = make_eval_fn(
                        model_arch, len(feature_cols), num_classes, test_loader, device
                    )

                    results = train_baselines(
                        strategies_to_run=strats,
                        clients=fl_clients,
                        num_rounds=HP["num_rounds"],
                        clients_per_round=HP["clients_per_round"],
                        num_classes=num_classes,
                        public_loader=public_loader,
                        seed=seed,
                        eval_fn=eval_fn,
                    )
                    type_seed_results[seed] = results
                    log("SEED_DONE", seed=seed, strategies=list(results.keys()))

                if not type_seed_results:
                    continue

                all_strategies = list(next(iter(type_seed_results.values())).keys())

                # --- Mean ± std across seeds ---
                print(f"\n{'='*70}")
                print(f"Dataset: {ds_name} | Model: {model_name} | Partition: {ptype}")
                print(f"{'='*70}")
                for strat in all_strategies:
                    accs = [
                        type_seed_results[s][strat]["final_results"].get("accuracy", 0.0)
                        for s in type_seed_results
                    ]
                    f1s = [
                        type_seed_results[s][strat]["final_results"].get("f1_macro", 0.0)
                        for s in type_seed_results
                    ]
                    print(f"  {strat:<14} acc={mean_std_str(accs)}  f1={mean_std_str(f1s)}")

                # --- Confusion matrices (last seed) ---
                last_seed    = sorted(type_seed_results.keys())[-1]
                last_results = type_seed_results[last_seed]

                for strat_name, obj in last_results.items():
                    fr     = obj.get("final_results", {})
                    y_true = fr.get("y_true", [])
                    y_pred = fr.get("y_pred", [])
                    if not y_true:
                        continue
                    title     = f"{ds_name} | {ptype} | {model_name} | {strat_name}"
                    fname     = _safe_stem(f"cm_{ds_name}_{ptype}_{model_name}_{strat_name}")
                    save_path = f"results/figures/{fname}.png"
                    src.eval.metrics.plot_confusion_matrix(
                        y_true, y_pred, class_names, title, save_path
                    )

                # --- Training curves (last seed) ---
                last_histories = {s: last_results[s]["history"] for s in last_results}
                curves_fname   = _safe_stem(f"curves_{ds_name}_{ptype}_{model_name}")
                plot_training_curves(last_histories, f"results/figures/{curves_fname}.png")

                # --- Comparison table (mean across seeds) ---
                table_data: Dict[str, dict] = {}
                for strat in all_strategies:
                    seed_frs = [
                        type_seed_results[s][strat]["final_results"]
                        for s in type_seed_results
                    ]
                    table_data[strat] = {
                        "accuracy":   float(np.mean([fr.get("accuracy", 0.0) for fr in seed_frs])),
                        "f1_macro":   float(np.mean([fr.get("f1_macro", 0.0) for fr in seed_frs])),
                        "f1_weighted":float(np.mean([fr.get("f1_weighted", 0.0) for fr in seed_frs])),
                        "rounds_to_threshold": [fr.get("rounds_to_threshold") for fr in seed_frs],
                        "total_bytes":float(np.mean([fr.get("total_bytes", 0) for fr in seed_frs])),
                    }
                table_fname = _safe_stem(f"table_{ds_name}_{ptype}_{model_name}")
                generate_comparison_table(table_data, f"results/tables/{table_fname}.txt")

        print(f"\n[DONE] Dataset: {ds_name}")

    print("\n" + "#" * 70)
    print("All experiments complete.")
    print("#" * 70)


if __name__ == "__main__":
    main()
