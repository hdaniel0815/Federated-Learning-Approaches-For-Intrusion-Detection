import copy
import torch
from .base import FederatedStrategy
from .base import ServerPayload, ClientUpdate


class FedProxAlgorithm(FederatedStrategy):
    name = "fedprox"

    def __init__(self, initial_weights: dict, mu: float = 0.01):
        self.initial_weights = initial_weights
        self.mu = mu

    def init_server_payload(self):
        # server state is global weights (same as FedAvg)
        return ServerPayload(weights=copy.deepcopy(self.initial_weights))

    def server_payload(self, global_model: ServerPayload):
        # broadcast weights (and optionally mu in extra fields if you add them)
        return global_model

    def aggregate(self, global_model: ServerPayload, client_updates: list[ClientUpdate]):
        # identical to FedAvg aggregation
        updates = [u for u in client_updates if u.weights is not None and u.n_samples > 0]
        if not updates:
            return global_model

        total = sum(u.n_samples for u in updates)

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
            pub_logits=global_model.pub_logits,
            prototypes=global_model.prototypes,
        )
    

class FedProxPlugin:
    def __init__(self, mu: float):
        self.mu = mu
        self.w0 = None

    def on_round_start(self, client, payload: ServerPayload):
        # snapshot model weights at start of round (after loading global weights)
        if self.mu > 0:
            self.w0 = {k: v.detach().clone() for k, v in client.model.state_dict().items()}

    def extra_loss(self, client, batch, forward_out, payload: ServerPayload):
        if self.mu <= 0 or self.w0 is None:
            return torch.tensor(0.0, device=client.device)

        cur = client.model.state_dict()
        prox = 0.0
        for k in cur.keys():
            prox = prox + (cur[k] - self.w0[k]).float().pow(2).sum()

        return (self.mu / 2.0) * prox

    def after_step(self, client, batch, forward_out, payload): 
        pass

    def on_round_end(self, client, payload):
        return {}