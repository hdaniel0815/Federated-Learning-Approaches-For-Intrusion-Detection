import json
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq

from src.fl.base import ServerPayload, ClientUpdate
from src.fl.fedavg import FedAvgAlgorithm, FedAvgPlugin
from src.fl.fedprox import FedProxAlgorithm, FedProxPlugin
from src.fl.fedmd import FedMDAlgorithm, FedMDPlugin

from src.models.mlp import StudentMLP
from src.models.lstm import StudentLSTM
from src.models.cnn1d import StudentCNN1D
from src.models.transformer import StudentTabTransformer

import src.eval.metrics

# federated training loop

# select clients, pick strategy, train locally, aggregate at server, broadcast, repeat

# initial loop through all the strategies, 

SEEDS = [42, 123, 456]

# FED_LEARNING_STRATS = [
#     "fedavg": lambda: FedAvgAlgorithm(),
#     "fedprox": lambda: FedProxAlgorithm(),
# ]

MODEL_BUILDERS = {
    # "mlp": lambda input_dim, num_classes: StudentMLP(input_dim, num_classes),
    # "cnn1d": lambda input_dim, num_classes: StudentCNN1D(input_dim, num_classes),
    "lstm": lambda input_dim, num_classes: StudentLSTM(input_dim, num_classes),
    "transformer": lambda input_dim, num_classes: StudentTabTransformer(input_dim, num_classes),
}

PATH_TO_TRAIN_PARTITIONS = "./partitions/train"
PATH_TO_TEST_PARTITIONS = "./partitions/test"
PATH_TO_PARQUETS = "./data/cic2018/processed_final"

STATS_DIR = "data/cic2018/stats"

with open(f"{STATS_DIR}/meta.json") as f:
    meta = json.load(f)

FEATURE_COLS = meta["num_cols"] + meta["cat_cols"]
LABEL_COL = meta["label_encoded_col"]
NUM_CLASSES = len(json.load(open(f"{STATS_DIR}/label_encoder.json")))

# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_std_str(values: List[float]) -> str:
    x = np.asarray(values, dtype=float)
    if len(x) <= 1:
        return f"{float(x.mean() if len(x) else 0.0):.4f} ± 0.0000"
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}"


def log(stage: str, **kv: Any) -> None:
    msg = " ".join([f"{k}={v}" for k, v in kv.items()])
    print(f"[{stage}] {msg}")

def test_parts_to_rows(parts_dir, test_parts):
    out = []
    for part in test_parts:
        # get row count cheaply (reads metadata + column)
        n = len(pd.read_parquet(Path(parts_dir) / part, columns=[LABEL_COL]))
        out.append({"part": part, "rows": list(range(n))})
    return out

# -----------------------------
# Dataset built from (part -> rows)
# -----------------------------
class ClientParquetRowsDataset(Dataset):
    """
    parts_to_rows: list of {"part": "final_part_00000.parquet", "rows": [1,2,3,...]}
    """
    def __init__(self, parts_dir: Path, parts_to_rows: List[dict], feature_cols: List[str], label_col: str):
        self.parts_dir = Path(parts_dir)
        self.feature_cols = list(feature_cols)
        self.label_col = label_col

        # normalize to: part_name -> np.array(rows)
        part_map: Dict[str, np.ndarray] = {}
        for item in parts_to_rows:
            part = item["part"]
            rows = np.array(item["rows"], dtype=np.int64)
            part_map[part] = rows
        self.part_to_rows = part_map

        # flatten to samples: (part, j) where j indexes into part_to_rows[part]
        self.samples: List[Tuple[str, int]] = []
        for part, rows in self.part_to_rows.items():
            for j in range(len(rows)):
                self.samples.append((part, j))

        self.samples.sort(key=lambda t: (t[0], int(self.part_to_rows[t[0]][t[1]])))

        # small per-worker cache to avoid re-reading parquet every row
        self._cache_part: Optional[str] = None
        self._cache_df: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_part(self, part: str):
        if self._cache_part == part and self._cache_df is not None:
            return self._cache_df
        df = pd.read_parquet(self.parts_dir / part, columns=self.feature_cols + [self.label_col])
        X = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = df[self.label_col].to_numpy()
        self._cache_part = part
        self._cache_df = (X, y)          # store arrays, not DF
        return self._cache_df

    def __getitem__(self, idx: int):
        part, j = self.samples[idx]
        row_idx = int(self.part_to_rows[part][j])

        X, y = self._load_part(part)
        x = torch.from_numpy(X[row_idx])
        yy = torch.tensor(int(y[row_idx]), dtype=torch.long)
        return x, yy
    
class PublicParquetRowsDataset(Dataset):
    def __init__(self, parts_dir: str | Path, samples, feature_cols, label_col):
        self.parts_dir = Path(parts_dir)
        self.samples = samples              # list[(part_name, row_idx)]
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self._cache_part = None
        self._cache_df = None

    def __len__(self):
        return len(self.samples)

    def _load_part(self, part_name: str) -> pd.DataFrame:
        if self._cache_part == part_name and self._cache_df is not None:
            return self._cache_df
        df = pd.read_parquet(self.parts_dir / part_name, columns=self.feature_cols + [self.label_col])
        self._cache_part, self._cache_df = part_name, df
        return df

    def __getitem__(self, idx):
        part_name, row_idx = self.samples[idx]
        df = self._load_part(part_name)
        row = df.iloc[int(row_idx)]
        x = torch.tensor(row[self.feature_cols].to_numpy(dtype=np.float32))
        y = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return x, y

def make_public_loader(
    *,
    parts_dir: str | Path,
    feature_cols: list[str],
    label_col: str,
    public_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
):
    parts_dir = Path(parts_dir)
    part_files = sorted([p.name for p in parts_dir.glob("final_part_*.parquet")])
    if not part_files:
        raise FileNotFoundError(f"No final_part_*.parquet found in {parts_dir}")

    # get num rows per part (fast, reads parquet metadata only)
    row_counts = np.array([pq.ParquetFile(parts_dir / f).metadata.num_rows for f in part_files], dtype=np.int64)
    probs = row_counts / row_counts.sum()

    rng = np.random.default_rng(seed)

    # sample which part each public sample comes from (proportional to rows)
    chosen_parts_idx = rng.choice(len(part_files), size=min(public_size, int(row_counts.sum())), replace=True, p=probs)

    samples = []
    for pi in chosen_parts_idx:
        nrows = int(row_counts[pi])
        ridx = int(rng.integers(0, nrows))
        samples.append((part_files[pi], ridx))
    samples.sort(key=lambda t: (t[0]))

    pub_ds = PublicParquetRowsDataset(parts_dir, samples, feature_cols, label_col)

    return DataLoader(
        pub_ds,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# -----------------------------
# Client
# -----------------------------
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
        mu: float = 0.0,               # FedProx
    ):
        self.cid = cid
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.public_loader = public_loader

        self.distill_lambda = distill_lambda
        self.distill_T = distill_T
        self.mu = mu

        # Option 1: strategy sets plugins once per run (recommended)
        self.plugins: List[Any] = []

    def set_plugins(self, plugins: List[Any]) -> None:
        self.plugins = plugins

    def local_train(self, payload: ServerPayload, *, local_epochs=1, lr=1e-3, weight_decay=0.0) -> ClientUpdate:
        self._apply_payload(payload)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        ce = torch.nn.CrossEntropyLoss()

        plugins = self.plugins
        for p in plugins:
            p.on_round_start(self, payload)

        total_seen = 0
        self.model.train()

        for _ in range(local_epochs):
            for x, y in tqdm(self.train_loader, desc="Batching Progress"):
                x, y = x.to(self.device), y.to(self.device)
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

        return ClientUpdate(
            client_id=self.cid,
            n_samples=total_seen,
            weights=self._get_weights_cpu(),
            pub_logits=extra.get("pub_logits"),
            prototypes=extra.get("prototypes"),
        )

    def _apply_payload(self, payload: ServerPayload) -> None:
        if payload.weights is not None:
            self.model.load_state_dict(payload.weights, strict=True)

    def _forward_pack(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # normalize heterogeneous model outputs:
        try:
            out = self.model(x, return_all=True)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                logits, protos, feats = out
                return logits, protos, feats
        except TypeError:
            pass

        logits = self.model(x)
        # fallback: no proto/features
        return logits, logits, logits

    def _get_weights_cpu(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}


# -----------------------------
# Manifest loading
# -----------------------------
def load_partition_manifest(path_to_manifest: Path) -> Tuple[Dict[int, List[dict]], int]:
    with open(path_to_manifest, "r") as f:
        manifest = json.load(f)

    partitions = manifest["partition"]
    num_clients = int(manifest["num_clients"])

    client_map: Dict[int, List[dict]] = {}
    for client_id, part_list in partitions.items():
        cid = int(client_id.split("_")[1])
        client_map[cid] = part_list

    return client_map, num_clients


def build_client_datasets(
    client_map: Dict[int, List[dict]],
    *,
    parts_dir: Path,
    feature_cols: List[str],
    label_col: str,
) -> Dict[int, Dataset]:
    out: Dict[int, Dataset] = {}
    for cid, part_list in client_map.items():
        out[cid] = ClientParquetRowsDataset(parts_dir, part_list, feature_cols, label_col)
    return out


# -----------------------------
# Federated loop
# -----------------------------
def run_federated_learning(
    *,
    strategy,
    clients: List[FLClient],
    init_server_payload: ServerPayload,
    num_rounds: int,
    clients_per_round: int,
    eval_fn=None,
    seed: int = 42,
) -> Tuple[ServerPayload, list]:
    rng = random.Random(seed)
    all_cids = list(range(len(clients)))

    global_state = init_server_payload
    history: List[dict] = []

    pbar = tqdm(range(1, num_rounds + 1), desc=strategy.__class__.__name__)
    for rnd in pbar:
        k = min(clients_per_round, len(all_cids))
        selected = rng.sample(all_cids, k)

        payload = strategy.server_payload(global_state)

        updates: List[ClientUpdate] = []
        for cid in selected:
            updates.append(clients[cid].local_train(payload))

        global_state = strategy.aggregate(global_state, updates)

        row = {"round": rnd, "selected": selected}
        if eval_fn is not None:
            row["metrics"] = eval_fn(global_state, clients, rnd)
            pbar.set_postfix(row["metrics"])
        history.append(row)

    return global_state, history


# -----------------------------
# Baseline runner
# -----------------------------
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
    # I randomly generate the initial weights based on the seed passed in 
    init_client_weights = [copy.deepcopy(c.model.state_dict()) for c in clients]

    template_weights = copy.deepcopy(init_client_weights[0])

    def init_payload(name: str) -> ServerPayload:
        if name in ("fedavg", "fedprox"):
            return ServerPayload(weights=copy.deepcopy(template_weights))
        if name == "fedmd":
            if public_loader is None:
                raise ValueError("FedMD requires public_loader.")
            n_pub = len(public_loader.dataset)
            return ServerPayload(pub_logits=np.zeros((n_pub, num_classes), dtype=np.float32))
        raise ValueError(f"Unknown strategy: {name}")

    # I dont know if I like this design wise
    strategy_builders = {
        "fedavg":  lambda: FedAvgAlgorithm(initial_weights=copy.deepcopy(template_weights)),
        "fedprox": lambda: FedProxAlgorithm(initial_weights=copy.deepcopy(template_weights), mu=0.01),
        "fedmd":   lambda: FedMDAlgorithm(n_pub=(len(public_loader.dataset) if public_loader else 0), num_classes=num_classes),
    }

    results: Dict[str, dict] = {}

    for name in strategies_to_run:
        # reset clients
        for i, c in enumerate(clients):
            c.model.load_state_dict(copy.deepcopy(init_client_weights[i]), strict=True)

        # i also don't know if I like this design wise
        if name == "fedavg":
            plugins = [FedAvgPlugin()]
        elif name == "fedprox":
            plugins = [FedAvgPlugin(), FedProxPlugin(mu=0.01)]
        elif name == "fedmd":
            plugins = [FedAvgPlugin(), FedMDPlugin(T=2.0, kd_lambda=1.0)]
        else:
            plugins = []

        for c in clients:
            c.set_plugins(plugins)

        strategy = strategy_builders[name]()
        init_server_payload = init_payload(name)

        server_state, history = run_federated_learning(
            strategy=strategy,
            clients=clients,
            init_server_payload=init_server_payload,
            num_rounds=num_rounds,
            clients_per_round=clients_per_round,
            eval_fn=eval_fn,
            seed=seed,
        )
        results[name] = {"server_state": server_state, "history": history}

    return results


# -----------------------------
# Main experiment driver
# -----------------------------
def main():
    if not FEATURE_COLS:
        raise RuntimeError("Set FEATURE_COLS in this file before running.")
    log("CONFIG", partitions=str(PATH_TO_TRAIN_PARTITIONS), parquets=str(PATH_TO_PARQUETS), seeds=SEEDS)

    HP = {
        "num_rounds": 1,
        "clients_per_round": 10,
        "batch_size": 1024,
        "lr": 1e-3,
    }
    strategies_to_run = ["fedmd"]

    test_parts_to_rows = test_parts_to_rows(PATH_TO_PARQUETS, PATH_TO_TEST_PARTITIONS)
    test_dataset = ClientParquetRowsDataset(PATH_TO_TEST_PARTITIONS, test_parts_to_rows, FEATURE_COLS, LABEL_COL)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    # metrics[(partition_file, strategy)][metric] -> [values over seeds]
    metrics: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    partition_files = sorted([p for p in Path(PATH_TO_TRAIN_PARTITIONS).iterdir() if p.suffix == ".json"])
    if not partition_files:
        raise RuntimeError(f"No partition JSON files found in {PATH_TO_TRAIN_PARTITIONS}")

    for part_path in partition_files:
        part_name = part_path.stem
        log("PARTITION_START", partition=part_name)

        for seed in SEEDS:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available:
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            log("SEED_START", partition=part_name, seed=seed)
            set_all_seeds(seed)

            client_map, num_clients = load_partition_manifest(part_path)
            log("MANIFEST", partition=part_name, seed=seed, num_clients=num_clients)

            client_datasets = build_client_datasets(
                client_map,
                parts_dir=PATH_TO_PARQUETS,
                feature_cols=FEATURE_COLS,
                label_col=LABEL_COL,
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

            # TODO: create public_loader for FedMD (must be shuffle=False)
            # This should be a separate partition 
            public_loader = make_public_loader(
                parts_dir=PATH_TO_PARQUETS,
                feature_cols=FEATURE_COLS,
                label_col=LABEL_COL,
                public_size=20000,
                batch_size=256,
                seed=seed
            )

            # for homogenous testing 
            for model_name, model_arch in MODEL_BUILDERS.items():
                # Build clients (same architecture)
                clients: List[FLClient] = []
                for cid in range(num_clients):
                    model = model_arch(input_dim=len(FEATURE_COLS), num_classes=NUM_CLASSES)  # <-- your model
                    clients.append(
                        FLClient(
                            cid=cid,
                            model=model,
                            train_loader=client_loaders[cid],
                            device=device,
                            public_loader=public_loader,
                        )
                    )

                # Final-only evaluation stub (replace later)
                def eval_fn(server_state: ServerPayload, clients: List[FLClient], round_idx: int):
                    if round_idx != HP["num_rounds"]:
                        return {}
                    return {"final_acc": 0.0}

                results = train_baselines(
                    strategies_to_run=strategies_to_run,
                    clients=clients,
                    num_rounds=HP["num_rounds"],
                    clients_per_round=HP["clients_per_round"],
                    num_classes=NUM_CLASSES,
                    public_loader=public_loader,
                    seed=seed,
                    eval_fn=eval_fn,
                )

                # collect metrics
                for strat_name, obj in results.items():
                    hist = obj["history"]
                    last = hist[-1].get("metrics", {}) if hist else {}
                    log("FINAL_METRICS", model_name=model_name, partition=part_name, seed=seed, strategy=strat_name, metrics=last)

                    key = (part_name, strat_name)
                    metrics.setdefault(key, {})
                    for m, v in last.items():
                        metrics[key].setdefault(m, []).append(float(v))

        log("PARTITION_DONE", partition=part_name)

    # report mean ± std per (partition, strategy)
    print("\n" + "#" * 90)
    print("FINAL RESULTS (mean ± std over seeds)")
    print("#" * 90)

    for (part_name, strat_name), m_dict in metrics.items():
        print(f"\npartition={part_name} strategy={strat_name}")
        for m, vals in m_dict.items():
            print(f"  {m}: {mean_std_str(vals)}")


if __name__ == "__main__":
    main()