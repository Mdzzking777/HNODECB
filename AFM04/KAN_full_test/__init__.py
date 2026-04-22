"""Isolated full functional test bed for pykan inside AFM04."""

from .config import KANFullTestConfig, default_config


def run_full_test(*args, **kwargs):
    from .train import run_full_test as _run_full_test

    return _run_full_test(*args, **kwargs)

__all__ = ["KANFullTestConfig", "default_config", "run_full_test"]
