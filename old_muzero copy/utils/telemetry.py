from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch


def append_metric(
    metrics: Dict[str, Any],
    key: str,
    value: Any,
    *,
    subkey: Optional[str] = None,
) -> None:
    metric_ops = metrics.setdefault("_ops", defaultdict(list))
    metric_ops[("append", key, subkey)].append(_to_metric_value(value))


def set_metric(
    metrics: Dict[str, Any],
    key: str,
    value: Any,
    *,
    subkey: Optional[str] = None,
) -> None:
    metric_sets = metrics.setdefault("_sets", {})
    metric_sets[(key, subkey)] = _to_metric_value(value)


def add_latent_visualization_metric(
    metrics: Dict[str, Any],
    key: str,
    latents: Any,
    *,
    labels: Any = None,
    method: str = "pca",
    **kwargs,
) -> None:
    metric_visualizations = metrics.setdefault("_latent_visualizations", {})
    metric_visualizations[key] = {
        "latents": _to_metric_value(latents),
        "labels": _to_metric_value(labels),
        "method": method,
        "kwargs": kwargs,
    }


def finalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    finalized: Dict[str, Any] = {}

    for (op, key, subkey), values in metrics.get("_ops", {}).items():
        if op != "append" or not values:
            continue
        aggregated = _aggregate_metric_values(values)
        _store_metric_value(finalized, key, aggregated, subkey=subkey)

    for (key, subkey), value in metrics.get("_sets", {}).items():
        _store_metric_value(finalized, key, value, subkey=subkey)

    latent_visualizations = metrics.get("_latent_visualizations")
    if latent_visualizations:
        finalized["_latent_visualizations"] = latent_visualizations

    return finalized


def _store_metric_value(
    metrics: Dict[str, Any],
    key: str,
    value: Any,
    *,
    subkey: Optional[str] = None,
) -> None:
    if subkey is None:
        metrics[key] = value
        return

    submetrics = metrics.setdefault(key, {})
    submetrics[subkey] = value


def _aggregate_metric_values(values: List[Any]) -> Any:
    if len(values) == 1:
        return values[0]

    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack([v.float() for v in values]).mean(dim=0)
    if isinstance(first, (int, float)):
        return sum(float(v) for v in values) / len(values)
    return values[-1]


def _to_metric_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu()
    return value
