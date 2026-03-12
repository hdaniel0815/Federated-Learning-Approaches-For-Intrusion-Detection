import numpy as np
from typing import Optional
from dataclasses import dataclass


class FederatedStrategy():

    def server_payload(self, global_model):
        pass

    def aggregate(self, global_model, client_updates):
        pass


@dataclass
class ClientUpdate:
    client_id: int
    n_samples: int
    weights: dict | None = None
    pub_logits: np.ndarray | None = None
    prototypes: dict | None = None
    bytes_sent: int = 0   # bytes communicated to server this round


@dataclass
class ServerPayload:
    weights: dict | None = None
    pub_logits: np.ndarray | None = None
    prototypes: dict | None = None


    