"""
Group 1 — FL algorithm correctness tests.

Covers:
  - ClientUpdate / ServerPayload dataclass defaults
  - FedAvgAlgorithm.aggregate() weighted average
  - FedProxPlugin.extra_loss() proximal term
  - FedProtoKDAlgorithm.aggregate() weights + prototype aggregation
  - FedProtoKDPlugin.extra_loss() with/without global prototypes
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.fl.base import ClientUpdate, ServerPayload
from src.fl.fedavg import FedAvgAlgorithm, FedAvgPlugin
from src.fl.fedprox import FedProxPlugin
from src.fl.fedprotokd import FedProtoKDAlgorithm, FedProtoKDPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weights(val: float) -> dict:
    """Minimal weight dict: one float tensor + one int tensor."""
    return {
        "w":   torch.full((2, 2), val, dtype=torch.float32),
        "idx": torch.tensor([0], dtype=torch.int64),
    }


def make_update(n_samples: int, val: float, **kwargs) -> ClientUpdate:
    return ClientUpdate(
        client_id=0,
        n_samples=n_samples,
        weights=make_weights(val),
        **kwargs,
    )


class SimpleModel(nn.Module):
    """Tiny 2-parameter model for plugin tests."""
    def __init__(self, init_val: float = 1.0):
        super().__init__()
        self.w = nn.Parameter(torch.full((2,), init_val))

    def forward(self, x, return_all=False):
        logits = x @ self.w
        if return_all:
            proj = self.w.unsqueeze(0).expand(x.size(0), -1)
            return logits, proj, proj
        return logits


class MockClient:
    def __init__(self, model: nn.Module):
        self.model  = model
        self.device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

def test_client_update_bytes_sent_default():
    u = ClientUpdate(client_id=0, n_samples=10, weights={"w": torch.zeros(2)})
    assert u.bytes_sent == 0


def test_client_update_optional_fields_none_by_default():
    u = ClientUpdate(client_id=1, n_samples=5)
    assert u.weights is None
    assert u.pub_logits is None
    assert u.prototypes is None


def test_server_payload_defaults():
    p = ServerPayload()
    assert p.weights is None
    assert p.pub_logits is None
    assert p.prototypes is None


# ---------------------------------------------------------------------------
# FedAvgAlgorithm
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_weighted_mean():
    """Weighted average: (val0*n0 + val1*n1) / (n0+n1)."""
    alg   = FedAvgAlgorithm(initial_weights=make_weights(0.0))
    state = ServerPayload(weights=make_weights(0.0))

    u0 = make_update(n_samples=1, val=2.0)
    u1 = make_update(n_samples=3, val=4.0)

    result   = alg.aggregate(state, [u0, u1])
    expected = (2.0 * 1 + 4.0 * 3) / 4   # = 3.5

    assert torch.allclose(
        result.weights["w"],
        torch.full((2, 2), expected),
        atol=1e-5,
    )


def test_fedavg_aggregate_single_client():
    """Single client: result weights == client weights."""
    alg   = FedAvgAlgorithm(initial_weights=make_weights(0.0))
    state = ServerPayload(weights=make_weights(0.0))
    u     = make_update(n_samples=10, val=7.0)

    result = alg.aggregate(state, [u])
    assert torch.allclose(result.weights["w"], torch.full((2, 2), 7.0), atol=1e-5)


def test_fedavg_aggregate_empty_returns_state():
    """No valid updates → return current server state unchanged."""
    alg   = FedAvgAlgorithm(initial_weights=make_weights(9.0))
    state = ServerPayload(weights=make_weights(9.0))

    result = alg.aggregate(state, [])
    assert torch.allclose(result.weights["w"], torch.full((2, 2), 9.0))


def test_fedavg_aggregate_preserves_int_tensor():
    """Integer tensors must not be averaged (copy first client's value)."""
    alg   = FedAvgAlgorithm(initial_weights=make_weights(0.0))
    state = ServerPayload(weights=make_weights(0.0))

    u0 = make_update(n_samples=1, val=1.0)
    u1 = make_update(n_samples=1, val=1.0)
    result = alg.aggregate(state, [u0, u1])

    assert result.weights["idx"].dtype == torch.int64


def test_fedavg_aggregate_skips_zero_sample_updates():
    """Updates with n_samples=0 should be ignored."""
    alg   = FedAvgAlgorithm(initial_weights=make_weights(0.0))
    state = ServerPayload(weights=make_weights(0.0))

    good = make_update(n_samples=4, val=8.0)
    bad  = make_update(n_samples=0, val=99.0)   # should be ignored

    result = alg.aggregate(state, [good, bad])
    assert torch.allclose(result.weights["w"], torch.full((2, 2), 8.0), atol=1e-5)


# ---------------------------------------------------------------------------
# FedProxPlugin
# ---------------------------------------------------------------------------

def test_fedprox_extra_loss_zero_before_round_start():
    """w0 is None before first on_round_start → loss must be 0."""
    plugin = FedProxPlugin(mu=0.1)
    model  = SimpleModel(init_val=1.0)
    client = MockClient(model)

    loss = plugin.extra_loss(client, None, None, ServerPayload())
    assert loss.item() == pytest.approx(0.0)


def test_fedprox_extra_loss_zero_when_unchanged(capsys):
    """Weights unchanged after snapshot → loss = 0."""
    plugin  = FedProxPlugin(mu=0.5)
    model   = SimpleModel(init_val=2.0)
    client  = MockClient(model)
    payload = ServerPayload(weights={"w": torch.ones(2)})

    plugin.on_round_start(client, payload)

    loss = plugin.extra_loss(client, None, None, payload)
    assert loss.item() == pytest.approx(0.0)


def test_fedprox_extra_loss_nonzero_after_perturbation(capsys):
    """Moving weights away from snapshot should give positive proximal loss."""
    plugin  = FedProxPlugin(mu=0.1)
    model   = SimpleModel(init_val=1.0)
    client  = MockClient(model)
    payload = ServerPayload(weights={"w": torch.ones(2)})

    plugin.on_round_start(client, payload)

    # Perturb model weights away from snapshot
    with torch.no_grad():
        model.w.copy_(torch.tensor([3.0, 3.0]))

    loss = plugin.extra_loss(client, None, None, payload)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# FedProtoKDAlgorithm
# ---------------------------------------------------------------------------

def test_fedprotokd_weight_aggregation_matches_fedavg():
    """FedProtoKD weight aggregation must equal FedAvg on identical inputs."""
    fedavg_alg = FedAvgAlgorithm(initial_weights=make_weights(0.0))
    proto_alg  = FedProtoKDAlgorithm(initial_weights=make_weights(0.0))
    state      = ServerPayload(weights=make_weights(0.0))

    u0 = make_update(n_samples=2, val=1.0)
    u1 = make_update(n_samples=6, val=3.0)

    r_avg   = fedavg_alg.aggregate(state, [u0, u1])
    r_proto = proto_alg.aggregate(state, [u0, u1])

    assert torch.allclose(r_avg.weights["w"], r_proto.weights["w"], atol=1e-5)


def test_fedprotokd_prototype_weighted_mean_per_class():
    """Global prototype = sample-weighted mean of client prototypes per class."""
    alg   = FedProtoKDAlgorithm(initial_weights=make_weights(0.0))
    state = ServerPayload(weights=make_weights(0.0))

    # class-0: client_0=[1,1] n=1, client_1=[3,3] n=3 → expected=[2.5, 2.5]
    u0 = ClientUpdate(
        client_id=0, n_samples=1,
        weights=make_weights(1.0),
        prototypes={0: np.array([1.0, 1.0], dtype=np.float32)},
    )
    u1 = ClientUpdate(
        client_id=1, n_samples=3,
        weights=make_weights(1.0),
        prototypes={0: np.array([3.0, 3.0], dtype=np.float32)},
    )

    result = alg.aggregate(state, [u0, u1])
    np.testing.assert_allclose(result.prototypes[0], [2.5, 2.5], atol=1e-5)


def test_fedprotokd_prototype_multi_class_partial():
    """Class missing from a client should still be aggregated from the other."""
    alg   = FedProtoKDAlgorithm(initial_weights=make_weights(0.0))
    state = ServerPayload(weights=make_weights(0.0))

    u0 = ClientUpdate(
        client_id=0, n_samples=1, weights=make_weights(1.0),
        prototypes={0: np.array([2.0], dtype=np.float32)},
    )
    u1 = ClientUpdate(
        client_id=1, n_samples=1, weights=make_weights(1.0),
        prototypes={
            0: np.array([4.0], dtype=np.float32),
            1: np.array([9.0], dtype=np.float32),
        },
    )

    result = alg.aggregate(state, [u0, u1])
    assert 0 in result.prototypes
    assert 1 in result.prototypes
    np.testing.assert_allclose(result.prototypes[0], [3.0], atol=1e-5)  # (2+4)/2
    np.testing.assert_allclose(result.prototypes[1], [9.0], atol=1e-5)  # only u1


def test_fedprotokd_no_prototypes_gives_none():
    """If no update includes prototypes, result.prototypes must be None."""
    alg     = FedProtoKDAlgorithm(initial_weights=make_weights(0.0))
    state   = ServerPayload(weights=make_weights(0.0))
    updates = [make_update(n_samples=5, val=1.0)]   # no prototypes

    result = alg.aggregate(state, updates)
    assert result.prototypes is None


def test_fedprotokd_init_server_payload():
    """init_server_payload should carry initial weights and no prototypes."""
    alg     = FedProtoKDAlgorithm(initial_weights=make_weights(5.0))
    payload = alg.init_server_payload()

    assert payload.weights is not None
    assert payload.prototypes is None
    assert torch.allclose(payload.weights["w"], torch.full((2, 2), 5.0))


# ---------------------------------------------------------------------------
# FedProtoKDPlugin
# ---------------------------------------------------------------------------

def test_fedprotokd_plugin_extra_loss_zero_on_first_round():
    """Round 1: payload.prototypes is None → loss must be exactly 0."""
    plugin      = FedProtoKDPlugin(alpha_proto=0.5)
    client      = MockClient(SimpleModel())
    payload     = ServerPayload(prototypes=None)
    x           = torch.randn(4, 2)
    y           = torch.tensor([0, 1, 0, 1])
    forward_out = (x, x, x)

    loss = plugin.extra_loss(client, (x, y), forward_out, payload)
    assert loss.item() == pytest.approx(0.0)


def test_fedprotokd_plugin_extra_loss_positive_with_mismatch():
    """Projection near zero vs global prototype at 10 → large positive loss."""
    plugin  = FedProtoKDPlugin(alpha_proto=1.0)
    client  = MockClient(SimpleModel())
    payload = ServerPayload(
        prototypes={0: np.array([10.0, 10.0], dtype=np.float32)}
    )

    protos      = torch.zeros(2, 2)       # [B=2, dim=2] — far from prototype
    x           = torch.zeros(2, 2)
    y           = torch.tensor([0, 0])    # both samples class 0
    forward_out = (x, protos, x)

    loss = plugin.extra_loss(client, (x, y), forward_out, payload)
    assert loss.item() > 0.0


def test_fedprotokd_plugin_on_round_end_returns_prototypes():
    """on_round_end must return a dict with 'prototypes' key."""
    plugin  = FedProtoKDPlugin(alpha_proto=0.3)

    # Build a tiny model and a tiny DataLoader
    model  = SimpleModel(init_val=1.0)
    client = MockClient(model)

    x = torch.randn(6, 2)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    from torch.utils.data import TensorDataset, DataLoader
    client.train_loader = DataLoader(TensorDataset(x, y), batch_size=3)

    result = plugin.on_round_end(client, ServerPayload())
    assert "prototypes" in result
    assert isinstance(result["prototypes"], dict)
    # Expect 3 classes
    assert set(result["prototypes"].keys()) == {0, 1, 2}
