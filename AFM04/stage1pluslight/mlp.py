"""AFM03-style MLP backend cloned into AFM04 stage1pluslight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def glorot_uniform(rng: np.random.Generator, fan_in: int, fan_out: int, shape: tuple[int, ...]) -> np.ndarray:
    limit = math.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(float)


def hidden_width_from_nodes(num_hidden_nodes: int) -> int:
    return 2 ** int(num_hidden_nodes)


@dataclass(frozen=True)
class DenseLayerParams:
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class MLPParams:
    layers: tuple[DenseLayerParams, ...]


def flatten_params(params: MLPParams) -> np.ndarray:
    flat: list[np.ndarray] = []
    for layer in params.layers:
        flat.append(np.asarray(layer.weight, dtype=float).reshape(-1))
        flat.append(np.asarray(layer.bias, dtype=float).reshape(-1))
    if not flat:
        return np.zeros(0, dtype=float)
    return np.concatenate(flat, dtype=float)


def unflatten_params(flat: np.ndarray, template: MLPParams) -> MLPParams:
    vec = np.asarray(flat, dtype=float).reshape(-1)
    cursor = 0
    layers: list[DenseLayerParams] = []
    for layer in template.layers:
        w_shape = layer.weight.shape
        b_shape = layer.bias.shape
        w_size = int(np.prod(w_shape))
        b_size = int(np.prod(b_shape))
        weight = vec[cursor : cursor + w_size].reshape(w_shape).copy()
        cursor += w_size
        bias = vec[cursor : cursor + b_size].reshape(b_shape).copy()
        cursor += b_size
        layers.append(DenseLayerParams(weight=weight, bias=bias))
    if cursor != vec.size:
        raise ValueError("flat parameter vector size does not match template")
    return MLPParams(layers=tuple(layers))


def build_mlp_params(seed: int, num_hidden_layers: int, num_hidden_nodes: int) -> MLPParams:
    if int(num_hidden_layers) < 0 or int(num_hidden_layers) > 2:
        raise ValueError("num_hidden_layers must lie in 0:2 to match AFM03")
    if int(num_hidden_nodes) < 1:
        raise ValueError("num_hidden_nodes must be >= 1")
    rng = np.random.default_rng(int(seed))
    hidden = hidden_width_from_nodes(num_hidden_nodes)
    dims = [(3, hidden)]
    for _ in range(int(num_hidden_layers)):
        dims.append((hidden, hidden))
    dims.append((hidden, 1))
    layers: list[DenseLayerParams] = []
    for fan_in, fan_out in dims:
        weight = glorot_uniform(rng, fan_in, fan_out, (fan_out, fan_in))
        bias = glorot_uniform(rng, fan_in, fan_out, (fan_out,))
        layers.append(DenseLayerParams(weight=weight, bias=bias))
    return MLPParams(layers=tuple(layers))


def mlp_forward(x: np.ndarray, params: MLPParams) -> float:
    h = np.asarray(x, dtype=float).reshape(-1)
    if h.size != 3:
        raise ValueError("AFM04 stage1pluslight MLP expects a 3-vector input")
    last = len(params.layers) - 1
    for idx, layer in enumerate(params.layers):
        h = layer.weight @ h + layer.bias
        if idx != last:
            h = gelu(h)
    return float(np.asarray(h, dtype=float).reshape(-1)[0])


def build_mlp_bundle(seed: int, num_hidden_layers: int, num_hidden_nodes: int) -> dict[str, Any]:
    params = build_mlp_params(seed, num_hidden_layers, num_hidden_nodes)

    def model(u: np.ndarray, model_params: MLPParams) -> float:
        return mlp_forward(u, model_params)

    return {
        "model": model,
        "model_params": params,
        "model_param_vec": flatten_params(params),
        "unflatten": lambda flat: unflatten_params(np.asarray(flat, dtype=float), params),
        "force_mode": "direct_contact_model",
        "backend_label": "mlp",
    }


__all__ = [
    "MLPParams",
    "build_mlp_bundle",
    "build_mlp_params",
    "flatten_params",
    "gelu",
    "hidden_width_from_nodes",
    "mlp_forward",
    "unflatten_params",
]
