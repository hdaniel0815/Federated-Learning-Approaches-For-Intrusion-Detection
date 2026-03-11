import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base import FederatedStrategy
from .base import ServerPayload, ClientUpdate


class FedMDAlgorithm(FederatedStrategy):
    name = "fedmd"

    def __init__(self, n_pub: int, num_classes: int):
        self.n_pub = n_pub
        self.num_classes = num_classes

    def init_server_payload(self):
        # teacher logits start at zeros (common baseline)
        return ServerPayload(
            pub_logits=np.zeros((self.n_pub, self.num_classes), dtype=np.float32)
        )

    def server_payload(self, global_model: ServerPayload):
        # broadcast teacher logits to clients
        return global_model

    # weighted aggregation
    def aggregate(self, global_model: ServerPayload, client_updates: list[ClientUpdate]):
        # aggregate client pub_logits -> new teacher pub_logits
        updates = [u for u in client_updates if u.pub_logits is not None and u.n_samples > 0]
        if not updates:
            return global_model

        total = sum(u.n_samples for u in updates)

        # weighted mean of logits across clients
        agg = np.zeros_like(updates[0].pub_logits, dtype=np.float32)
        for u in updates:
            frac = u.n_samples / total
            agg += u.pub_logits.astype(np.float32) * frac

        return ServerPayload(
            weights=global_model.weights,          # usually None in FedMD
            pub_logits=agg,
            prototypes=global_model.prototypes,    # usually None
        )
    

class FedMDPlugin:
    def __init__(self, T: float = 2.0, kd_lambda: float = 1.0):
        self.T = T
        self.kd_lambda = kd_lambda

    def on_round_start(self, client, payload: ServerPayload):
        pass

    def extra_loss(self, client, batch, forward_out, payload: ServerPayload):
        # FedMD distillation is usually on PUBLIC data, not private batches,
        # so we keep this as no-op here and do KD in on_round_end (or a separate phase).
        return torch.tensor(0.0, device=client.device)

    def after_step(self, client, batch, forward_out, payload):
        pass

    def on_round_end(self, client, payload: ServerPayload):
        if payload.pub_logits is None:
            return {}
        if client.public_loader is None:
            raise RuntimeError("FedMDPlugin requires client.public_loader")

        # 1) Distill on public data using teacher logits from server payload
        teacher = torch.from_numpy(payload.pub_logits).to(client.device).float()
        T = self.T

        client.model.train()
        opt = torch.optim.Adam(client.model.parameters(), lr=1e-3)

        offset = 0
        for x_pub, _ in client.public_loader:
            x_pub = x_pub.to(client.device)
            b = x_pub.size(0)
            t = teacher[offset:offset + b]
            offset += b

            opt.zero_grad()
            out = client.model(x_pub)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.softmax(t / T, dim=1),
                reduction="batchmean",
            ) * (T * T)

            (self.kd_lambda * kd).backward()
            opt.step()

        # 2) Return this client’s logits on the public set (for server aggregation)
        client.model.eval()
        outs = []
        with torch.no_grad():
            for x_pub, _ in tqdm(client.public_loader, desc="Public Loader Training"):
                x_pub = x_pub.to(client.device)
                out = client.model(x_pub)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                outs.append(logits.detach())
        pub_logits_out = torch.cat(outs, dim=0).cpu().numpy().astype(np.float32)

        return {"pub_logits": pub_logits_out}