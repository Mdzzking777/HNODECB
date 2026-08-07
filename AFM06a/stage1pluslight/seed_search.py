"""Seed-only search plan replacing the AFM04 ks/cs mechanical grid."""

from __future__ import annotations

from dataclasses import dataclass


def nn_bank_seed(run_seed: int, seed_bank_index: int) -> int:
    """Use the same base-seed-plus-index mapping as AFM04 st1pl."""

    seed = int(run_seed) + int(seed_bank_index)
    if seed <= 0:
        raise ValueError("run seed plus bank index must be positive")
    return seed


@dataclass(frozen=True)
class SeedTrial:
    global_trial_id: int
    nn_seed_bank_idx: int
    nn_init_seed: int
    trial_label: str


def total_trials(nn_seed_bank_size: int) -> int:
    return max(0, int(nn_seed_bank_size))


def decode_seed_trial(global_trial_id: int, nn_seed_bank_size: int, run_seed: int) -> SeedTrial:
    total = total_trials(nn_seed_bank_size)
    if global_trial_id < 1 or global_trial_id > total:
        raise ValueError("global_trial_id out of range")
    index = int(global_trial_id)
    return SeedTrial(
        global_trial_id=index,
        nn_seed_bank_idx=index,
        nn_init_seed=nn_bank_seed(run_seed, index),
        trial_label=f"seed_{index:06d}",
    )


def compute_shard_assignments(total: int, shard_index: int, shard_count: int) -> list[int]:
    if shard_count < 1 or not (1 <= shard_index <= shard_count):
        raise ValueError("shard_index must be in 1..shard_count")
    return [trial_id for trial_id in range(1, int(total) + 1) if (trial_id - shard_index) % shard_count == 0]


__all__ = [
    "SeedTrial",
    "compute_shard_assignments",
    "decode_seed_trial",
    "nn_bank_seed",
    "total_trials",
]
