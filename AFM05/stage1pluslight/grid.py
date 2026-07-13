"""Grid-search helpers for AFM05 stage1pluslight."""

from __future__ import annotations

import math
from dataclasses import dataclass


# AFM05 experimental data has no known true ks/cs. These bounds are
# 05experimentPS mechanistic prior bounds, not a GT-centered range.
KS_BOUNDS = (0.1, 100.0)
CS_BOUNDS = (5.0e-9, 5.0e-5)


@dataclass(frozen=True)
class GridTrial:
    global_trial_id: int
    ks_node_idx: int
    cs_node_idx: int
    nn_seed_bank_idx: int
    ks0: float
    cs0: float
    node_label: str


def loggrid_value(node_idx: int, node_count: int, lo: float, hi: float) -> float:
    if node_count <= 1:
        return math.sqrt(lo * hi)
    alpha = float(node_idx) / float(node_count - 1)
    return math.exp(math.log(lo) + alpha * (math.log(hi) - math.log(lo)))


def total_trials(ks_node_count: int, cs_node_count: int, nn_seed_bank_size: int) -> int:
    return int(ks_node_count) * int(cs_node_count) * int(nn_seed_bank_size)


def decode_grid_trial(global_trial_id: int, ks_node_count: int, cs_node_count: int, nn_seed_bank_size: int) -> GridTrial:
    total = total_trials(ks_node_count, cs_node_count, nn_seed_bank_size)
    if global_trial_id < 1 or global_trial_id > total:
        raise ValueError("global_trial_id out of range")

    trial0 = global_trial_id - 1
    nn_seed_bank_idx = (trial0 % nn_seed_bank_size) + 1
    rem0 = trial0 // nn_seed_bank_size
    cs_node_idx = (rem0 % cs_node_count) + 1
    ks_node_idx = (rem0 // cs_node_count) + 1

    ks0 = loggrid_value(ks_node_idx - 1, ks_node_count, KS_BOUNDS[0], KS_BOUNDS[1])
    cs0 = loggrid_value(cs_node_idx - 1, cs_node_count, CS_BOUNDS[0], CS_BOUNDS[1])
    return GridTrial(
        global_trial_id=global_trial_id,
        ks_node_idx=ks_node_idx,
        cs_node_idx=cs_node_idx,
        nn_seed_bank_idx=nn_seed_bank_idx,
        ks0=ks0,
        cs0=cs0,
        node_label=f"node_{ks_node_idx}*{cs_node_idx}",
    )


def compute_shard_assignments(total: int, shard_index: int, shard_count: int) -> list[int]:
    if shard_index < 1 or shard_index > shard_count:
        raise ValueError("shard_index must be in 1..shard_count")
    return [i for i in range(1, total + 1) if ((i - shard_index) % shard_count) == 0]


__all__ = [
    "CS_BOUNDS",
    "GridTrial",
    "KS_BOUNDS",
    "compute_shard_assignments",
    "decode_grid_trial",
    "loggrid_value",
    "total_trials",
]
