from __future__ import annotations
import copy
import torch
from .base import FederatedStrategy
from .base import ServerPayload, ClientUpdate
from dataclasses import dataclass
from typing import Dict, List, Any


# take in a client data model, and take only what I need from this model on this level, the training loop should remain very independent

# strategy has context of what the model needs
class FedAvgAlgorithm(FederatedStrategy):
    name = "fedavg"

    def __init__(self, initial_weights: dict):
        self.initial_weights = initial_weights

    def init_server_payload(self):
        # server state is global weights (same as FedAvg)
        return ServerPayload(weights=copy.deepcopy(self.initial_weights))


    def aggregate(self, server_state: ServerPayload, updates: list[ClientUpdate]):
        updates = [u for u in updates if u.weights is not None and u.n_samples > 0]
        if not updates:
            return server_state
        
        total = sum(cli.n_samples for cli in updates)
        w_avg = copy.deepcopy(updates[0].weights)

        for k in w_avg.keys():
            if not torch.is_floating_point(w_avg[k]):
                w_avg[k] = updates[0].weights[k]
                continue
            w_avg[k] = torch.zeros_like(w_avg[k])

        for u in updates:
            w = {k: v.detach().cpu() for k, v in u.weights.items()}
            frac = u.n_samples / total
            for k in w_avg.keys():
                if torch.is_floating_point(w_avg[k]):
                    w_avg[k] += w[k] * frac

        return ServerPayload(
            weights=w_avg,
            pub_logits=server_state.pub_logits,
            prototypes=server_state.prototypes,
        )
    
    def server_payload(self, payload: ServerPayload):
        return payload
    

class FedAvgPlugin:
    def on_round_start(self, client, payload):
        # FedAvg does not add extra local losses or extra phases.
        pass

    def extra_loss(self, client, batch, forward_out, payload):
        # No extra loss term.
        return torch.tensor(0.0, device=client.device)

    def after_step(self, client, batch, forward_out, payload):
        pass

    def on_round_end(self, client, payload):
        # No extra outputs beyond weights; base loop returns weights anyway.
        return {}
