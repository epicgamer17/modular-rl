"""
PPO specific loss operators.
"""

# TODO: very algorithm specific filename, maybe update at some point

import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.registry import OperatorSpec, PortSpec, Scalar, TensorSpec

PPO_OPTIMIZER_SPEC = OperatorSpec.create(
    name="PPO_Optimizer",
    inputs={"loss": PortSpec(spec=None)},  # Can be dict or tensor
    outputs={},
    pure=False,
    stateful=True,
    allowed_contexts={"learner"},
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=True,
    math_category="optimizer",
)

VALUE_LOSS_SPEC = OperatorSpec.create(
    name="ValueLoss",
    inputs={
        "values": TensorSpec(shape=(-1,), dtype="float32"),
        "returns": TensorSpec(shape=(-1,), dtype="float32"),
        "old_values": PortSpec(
            spec=TensorSpec(shape=(-1,), dtype="float32"), required=False
        ),
    },
    outputs={"loss": TensorSpec(shape=(), dtype="float32")},
    allowed_contexts={"learner"},
    differentiable=True,
    creates_grad=True,
    consumes_grad=False,
    updates_params=False,
    math_category="loss",
)

SURROGATE_LOSS_SPEC = OperatorSpec.create(
    name="SurrogateLoss",
    inputs={
        "ratio": TensorSpec(shape=(-1,), dtype="float32"),
        "clipped_ratio": TensorSpec(shape=(-1,), dtype="float32"),
        "advantages": TensorSpec(shape=(-1,), dtype="float32"),
    },
    outputs={"loss": TensorSpec(shape=(), dtype="float32")},
    allowed_contexts={"learner"},
    differentiable=True,
    creates_grad=True,
    consumes_grad=False,
    updates_params=False,
    domain_tags={"policy_gradient"},
    math_category="loss",
)

# Implementation stubs if needed, or just specs if implemented elsewhere
