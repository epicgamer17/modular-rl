"""
Blackboard Diffing — per-component change tracking for RL debugging.

Captures a lightweight fingerprint of the blackboard before and after each
component executes, then produces a structured diff showing exactly what
was added, overwritten, or removed.

For tensors, the fingerprint includes shape, dtype, and object identity
(to detect overwrites without comparing contents). Optional tensor
statistics (mean, std, min, max, has_nan) are captured for overwritten
tensors to surface value explosions, collapses, and NaN propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.blackboard import Blackboard


# ──────────────────────────────────────────────────────────────────────
# Fingerprint
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValueFingerprint:
    """Lightweight descriptor of a single blackboard value."""
    obj_id: int
    type_name: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None

    @staticmethod
    def from_value(value: Any) -> ValueFingerprint:
        if torch.is_tensor(value):
            return ValueFingerprint(
                obj_id=id(value),
                type_name="Tensor",
                shape=tuple(value.shape),
                dtype=str(value.dtype),
            )
        return ValueFingerprint(
            obj_id=id(value),
            type_name=type(value).__name__,
        )


# ──────────────────────────────────────────────────────────────────────
# Snapshot
# ──────────────────────────────────────────────────────────────────────

# Type alias: {qualified_path: fingerprint}
BlackboardSnapshot = Dict[str, ValueFingerprint]

_SECTIONS = ("data", "predictions", "targets", "losses", "meta")


def snapshot_blackboard(blackboard: Blackboard) -> BlackboardSnapshot:
    """Capture a shallow fingerprint of every path on the blackboard."""
    snap: BlackboardSnapshot = {}
    for section_name in _SECTIONS:
        section = getattr(blackboard, section_name)
        _flatten_into(snap, section_name, section)
    return snap


def _flatten_into(
    out: BlackboardSnapshot,
    prefix: str,
    obj: Any,
) -> None:
    """Recursively flatten a nested dict into dotted-path fingerprints."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}"
            if isinstance(value, dict):
                _flatten_into(out, path, value)
            else:
                out[path] = ValueFingerprint.from_value(value)
    else:
        out[prefix] = ValueFingerprint.from_value(obj)


# ──────────────────────────────────────────────────────────────────────
# Tensor statistics (for overwritten tensors)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TensorStats:
    """Quick health summary of a tensor value."""
    shape: Tuple[int, ...]
    dtype: str
    mean: float
    std: float
    min: float
    max: float
    has_nan: bool
    has_inf: bool

    @staticmethod
    def from_tensor(t: torch.Tensor) -> TensorStats:
        with torch.no_grad():
            ft = t.float()
            return TensorStats(
                shape=tuple(t.shape),
                dtype=str(t.dtype),
                mean=ft.mean().item(),
                std=ft.std().item() if ft.numel() > 1 else 0.0,
                min=ft.min().item(),
                max=ft.max().item(),
                has_nan=bool(torch.isnan(ft).any()),
                has_inf=bool(torch.isinf(ft).any()),
            )


# ──────────────────────────────────────────────────────────────────────
# Diff
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OverwriteDetail:
    """Details about a single overwritten path."""
    path: str
    before: ValueFingerprint
    after: ValueFingerprint
    tensor_stats: Optional[TensorStats] = None


@dataclass(frozen=True)
class BlackboardDiff:
    """Structured diff between two blackboard snapshots."""
    component_name: str
    added: Tuple[str, ...]
    removed: Tuple[str, ...]
    overwrites: Tuple[OverwriteDetail, ...]

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed and not self.overwrites

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = []
        if self.added:
            parts.append(f"+{len(self.added)} added")
        if self.overwrites:
            parts.append(f"~{len(self.overwrites)} overwritten")
        if self.removed:
            parts.append(f"-{len(self.removed)} removed")
        body = ", ".join(parts) if parts else "no changes"
        return f"[{self.component_name}] {body}"

    def detail(self) -> str:
        """Multi-line detailed report."""
        lines = [self.summary()]
        for path in self.added:
            lines.append(f"  + {path}")
        for ow in self.overwrites:
            line = f"  ~ {ow.path}"
            if ow.before.shape != ow.after.shape:
                line += f"  shape: {ow.before.shape} -> {ow.after.shape}"
            if ow.before.dtype != ow.after.dtype:
                line += f"  dtype: {ow.before.dtype} -> {ow.after.dtype}"
            if ow.tensor_stats:
                s = ow.tensor_stats
                line += f"  mean={s.mean:.4g} std={s.std:.4g} [{s.min:.4g}, {s.max:.4g}]"
                if s.has_nan:
                    line += " NaN!"
                if s.has_inf:
                    line += " Inf!"
            lines.append(line)
        for path in self.removed:
            lines.append(f"  - {path}")
        return "\n".join(lines)


def diff_snapshots(
    component_name: str,
    before: BlackboardSnapshot,
    after: BlackboardSnapshot,
    blackboard: Blackboard,
) -> BlackboardDiff:
    """Compute the diff between two blackboard snapshots.

    Args:
        component_name: Name of the component that ran between snapshots.
        before:         Snapshot taken before execution.
        after:          Snapshot taken after execution.
        blackboard:     The live blackboard (used to pull tensor stats
                        for overwritten values).
    """
    before_paths = set(before)
    after_paths = set(after)

    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)

    overwrites: List[OverwriteDetail] = []
    for path in sorted(before_paths & after_paths):
        fp_before = before[path]
        fp_after = after[path]
        if fp_before.obj_id != fp_after.obj_id:
            # Value object changed — this is an overwrite
            stats = None
            if fp_after.type_name == "Tensor":
                try:
                    val = _resolve_path(blackboard, path)
                    if torch.is_tensor(val):
                        stats = TensorStats.from_tensor(val)
                except (KeyError, AttributeError):
                    pass
            overwrites.append(OverwriteDetail(
                path=path,
                before=fp_before,
                after=fp_after,
                tensor_stats=stats,
            ))

    return BlackboardDiff(
        component_name=component_name,
        added=tuple(added),
        removed=tuple(removed),
        overwrites=tuple(overwrites),
    )


def _resolve_path(blackboard: Blackboard, path: str) -> Any:
    """Resolve a qualified dotted path on the blackboard."""
    parts = path.split(".")
    container = getattr(blackboard, parts[0])
    current = container
    for key in parts[1:]:
        current = current[key]
    return current
