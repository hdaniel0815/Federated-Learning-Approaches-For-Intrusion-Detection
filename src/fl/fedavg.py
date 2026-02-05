import copy
import torch

@torch.no_grad()
def fedavg_aggregate(client_state_dicts, client_num_samples):
    """
    client_state_dicts: list[dict[str, Tensor]]  (one per client)
    client_num_samples: list[int]               (n_k per client)

    returns: averaged_state_dict
    """
    total = sum(client_num_samples)
    avg = copy.deepcopy(client_state_dicts[0])

    for k in avg.keys():
        # initialize accumulator
        avg[k] = torch.zeros_like(avg[k])
        # weighted sum
        for sd, n in zip(client_state_dicts, client_num_samples):
            avg[k] += sd[k] * (n / total)

    return avg
