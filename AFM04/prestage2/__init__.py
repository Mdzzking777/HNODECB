"""AFM04 prestage2 package."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Prestage2Config": ("AFM04.prestage2.config", "Prestage2Config"),
    "default_config": ("AFM04.prestage2.config", "default_config"),
    "merge_prestage2_results": ("AFM04.prestage2.runner", "merge_prestage2_results"),
    "run_driver": ("AFM04.prestage2.runner", "run_driver"),
    "run_prestage2_shard": ("AFM04.prestage2.runner", "run_prestage2_shard"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
