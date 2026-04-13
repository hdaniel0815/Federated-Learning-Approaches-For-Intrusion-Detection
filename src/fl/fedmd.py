"""
Round flow
──────────
1. Server → clients : consensus logits  (no weights)
2. Each client       : trains on private data + KL-distills from consensus
3. Clients → server  : each client's logits on the public dataset
4. Server            : weighted-averages those logits → new consensus
"""

import numpy as np
import torch
import torch.nn.functional as F
from .base import FederatedStrategy, ServerPayload, ClientUpdate
from typing import Optional


# ───────────────────────────────────────────────────────────────────
# Server-side strategy
# ───────────────────────────────────────────────────────────────────

class FedMDAlgorithm(FederatedStrategy):
    name = "fedmd"

    def __init__(self, model_arch, input_dim, num_classes, public_loader, device):
        self.public_loader = public_loader
        self.device = device
        # Kept for init_server_payload seed logits
        self._model_arch = model_arch
        self._input_dim = input_dim
        self._num_classes = num_classes

    def init_server_payload(self):
        """Seed consensus with logits from a random model (before any training)."""
        tmp = self._model_arch(self._input_dim, self._num_classes).to(self.device)
        tmp.eval()
        logits_list = []
        with torch.no_grad():
            for x, _ in self.public_loader:
                logits_list.append(tmp(x.to(self.device)).cpu())
        init_logits = torch.cat(logits_list, dim=0).numpy().astype(np.float32)
        # weights=None → clients keep their own local weights
        return ServerPayload(weights=None, pub_logits=init_logits)

    def server_payload(self, global_state: ServerPayload) -> ServerPayload:
        """Broadcast consensus logits only — no weights."""
        return ServerPayload(weights=None, pub_logits=global_state.pub_logits)

    def aggregate(
        self, global_state: ServerPayload, client_updates: list[ClientUpdate]
    ) -> ServerPayload:
        """
        Weighted average of the logit arrays each client computed on the
        public dataset.  No weight aggregation — that's the whole point
        of FedMD (enables heterogeneous client architectures).
        """
        valid = [u for u in client_updates if u.pub_logits is not None and u.n_samples > 0]
        if not valid:
            return global_state

        total = sum(u.n_samples for u in valid)
        consensus = np.zeros_like(valid[0].pub_logits, dtype=np.float64)
        for u in valid:
            consensus += np.array(u.pub_logits, dtype=np.float64) * (u.n_samples / total)

        return ServerPayload(
            weights=None,
            pub_logits=consensus.astype(np.float32),
        )


# ───────────────────────────────────────────────────────────────────
# Client-side plugin
# ───────────────────────────────────────────────────────────────────

class FedMDPlugin:
    """
    Client-side behaviour for FedMD:
      • on_round_start : cache the consensus logits
      • extra_loss     : KL divergence between student logits and consensus
      • on_round_end   : compute this client's logits on the public dataset
                         and return them so they're sent to the server
    """

    def __init__(self, T: float = 2.0, kd_lambda: float = 1.0):
        self.T = T
        self.kd_lambda = kd_lambda
        self._pub_logits: Optional[torch.Tensor] = None
        self._pub_index = 0

    # ── hooks ──────────────────────────────────────────────────────

    def on_round_start(self, client, payload: ServerPayload) -> None:
        if payload.pub_logits is not None:
            self._pub_logits = torch.tensor(
                payload.pub_logits, dtype=torch.float32,
            ).to(client.device)
        self._pub_index = 0

    def extra_loss(self, client, batch, outputs, payload) -> torch.Tensor:
        """KL(student || consensus) on a slice of the public logits."""
        if self._pub_logits is None:
            return torch.tensor(0.0, device=client.device)

        logits, _, _ = outputs
        bsz = logits.size(0)

        # Slide a window through the cached public logits so each
        # private-data batch gets a different slice of the consensus.
        end = min(self._pub_index + bsz, self._pub_logits.size(0))
        target_logits = self._pub_logits[self._pub_index:end]
        self._pub_index = end % self._pub_logits.size(0)

        n = min(bsz, target_logits.size(0))
        loss = F.kl_div(
            F.log_softmax(logits[:n] / self.T, dim=-1),
            F.softmax(target_logits[:n] / self.T, dim=-1),
            reduction="batchmean",
        ) * (self.T ** 2)

        return self.kd_lambda * loss

    def after_step(self, client, batch, forward_out, payload: ServerPayload):
        pass

    def on_round_end(self, client, payload: ServerPayload) -> dict:
        """
        Compute this client's logits on the public dataset.
        This is the ONLY thing that crosses the wire in FedMD.
        """
        if client.public_loader is None:
            return {}

        client.model.eval()
        all_logits = []
        with torch.no_grad():
            for x, _ in client.public_loader:
                x = x.to(client.device)
                logits = client.model(x)
                all_logits.append(logits.cpu())

        pub_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
        return {"pub_logits": pub_logits}
