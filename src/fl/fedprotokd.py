from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F
from .base import FederatedStrategy, ServerPayload, ClientUpdate


class FedProtoKDAlgorithm(FederatedStrategy):

    # I should use composition to add the plugin in this object
    name = "fedprotokd"

    def __init__(self, initial_weights: dict):
        self.initial_weights = initial_weights

    def init_server_payload(self):
        return ServerPayload(weights=copy.deepcopy(self.initial_weights), prototypes=None)

    def server_payload(self, state: ServerPayload) -> ServerPayload:
        # Broadcast both weights and current global prototypes
        return state

    def aggregate(self, state: ServerPayload, updates: list[ClientUpdate]) -> ServerPayload:
        # --- Weight aggregation (FedAvg-style) ---
        w_updates = [u for u in updates if u.weights is not None and u.n_samples > 0]
        if not w_updates:
            w_avg = state.weights
        else:
            total = sum(u.n_samples for u in w_updates)
            w_avg = copy.deepcopy(w_updates[0].weights)

            for k in w_avg.keys():
                if not torch.is_floating_point(w_avg[k]):
                    w_avg[k] = w_updates[0].weights[k]
                    continue
                w_avg[k] = torch.zeros_like(w_avg[k])

            for u in w_updates:
                frac = u.n_samples / total
                for k in w_avg.keys():
                    if torch.is_floating_point(w_avg[k]):
                        w_avg[k] += u.weights[k].detach().cpu() * frac

        # --- Prototype aggregation (weighted mean per class) ---
        p_updates = [u for u in updates if u.prototypes is not None and u.n_samples > 0]
        global_prototypes = None
        if p_updates:
            p_total = sum(u.n_samples for u in p_updates)
            all_classes = set(cls for u in p_updates for cls in u.prototypes)
            global_prototypes = {}
            for cls in all_classes:
                valid = [
                    (u.prototypes[cls], u.n_samples)
                    for u in p_updates
                    if cls in u.prototypes and u.prototypes[cls] is not None
                ]
                if valid:
                    tw = sum(w for _, w in valid)
                    agg = np.zeros_like(valid[0][0], dtype=np.float64)
                    for proto, w in valid:
                        agg += np.array(proto, dtype=np.float64) * (w / tw)
                    global_prototypes[cls] = agg.astype(np.float32)

        return ServerPayload(weights=w_avg, prototypes=global_prototypes)


class FedProtoKDPlugin:
    def __init__(self, alpha_proto: float = 0.3):
        self.alpha_proto = alpha_proto

    def on_round_start(self, client, payload: ServerPayload):
        pass

    def extra_loss(self, client, batch, forward_out, payload: ServerPayload):
        """MSE alignment between projection_head output and global class prototype."""
        if payload.prototypes is None:
            return torch.tensor(0.0, device=client.device)

        x, y = batch
        # forward_out = (logits, prototypes_proj, features) from _forward_pack / return_all=True
        _, protos, _ = forward_out

        loss_proto = torch.tensor(0.0, device=client.device)
        count = 0
        for i in range(len(y)):
            cls = int(y[i].item())
            if cls in payload.prototypes and payload.prototypes[cls] is not None:
                gp = torch.tensor(
                    payload.prototypes[cls], dtype=torch.float32, device=client.device
                )
                loss_proto = loss_proto + F.mse_loss(protos[i], gp)
                count += 1

        if count > 0:
            loss_proto = loss_proto / count
        return self.alpha_proto * loss_proto

    def after_step(self, client, batch, forward_out, payload: ServerPayload):
        pass

    def on_round_end(self, client, payload: ServerPayload) -> dict:
        """Compute local class prototypes (mean projection_head embeddings per class)."""
        client.model.eval()
        class_feats: dict[int, list] = {}

        with torch.no_grad():
            for x, y in client.train_loader:
                x = x.to(client.device)
                out = client.model(x, return_all=True)
                # out = (logits, projection_head_output, lstm_features)
                proj = out[1] if isinstance(out, (tuple, list)) else out
                for i, cls in enumerate(y.cpu().numpy()):
                    cls = int(cls)
                    class_feats.setdefault(cls, []).append(proj[i].cpu().numpy())

        local_prototypes = {
            cls: np.mean(feats, axis=0).astype(np.float32)
            for cls, feats in class_feats.items()
            if feats
        }
        return {"prototypes": local_prototypes}
