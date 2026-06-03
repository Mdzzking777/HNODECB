"""Isolated full functional test bed for MLP inside AFM04."""

from .config import MLPFullTestConfig, default_config


def run_full_test(*args, **kwargs):
    from .train import run_full_test as _run_full_test

    return _run_full_test(*args, **kwargs)

__all__ = ["MLPFullTestConfig", "default_config", "run_full_test"]
