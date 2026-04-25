from runtime.registry import (
    OperatorSpec,
    PortSpec,
    register_spec,
    get_spec,
    Tensor,
    Scalar,
    SingleObs,
    SingleQ,
    BatchObs,
    BatchQ,
    TransitionBatch,
)


def register_dqn_specs():
    """
    Registers specifications for all DQN-specific operators.
    These are used by the compiler for validation and shape inference.
    """

    # --- Built-in Specs (Required for strict compilation) ---
    register_spec(
        "Source",
        OperatorSpec.create(
            "Source",
            inputs={},
            outputs={},
            pure=True,
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )
    register_spec(
        "ReplayQuery",
        OperatorSpec.create(
            "ReplayQuery",
            inputs={},
            outputs={"default": TransitionBatch},
            pure=False,
            stateful=True,
            allowed_contexts={"learner"},
            requires_buffers=["main"],
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="buffer_io"
        ),
    )
    register_spec(
        "TargetSync",
        OperatorSpec.create(
            "TargetSync",
            inputs={},
            outputs={},
            pure=False,
            stateful=True,
            allowed_contexts={"learner"},
            side_effects=["target_update"],
            requires_models=["source", "target"],
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="control"
        ),
    )
    register_spec(
        "MetricsSink",
        OperatorSpec.create(
            "MetricsSink",
            inputs={
                "loss": PortSpec(spec=Scalar("float32"), required=False),
                "avg_q": PortSpec(spec=Scalar("float32"), required=False),
                "reward": PortSpec(spec=Scalar("float32"), required=False),
                "epsilon": PortSpec(spec=Scalar("float32"), required=False),
                "replay_size": PortSpec(spec=Scalar("int64"), required=False),
                "batch": PortSpec(spec=TransitionBatch, required=False),
                "default": PortSpec(
                    spec=Scalar("float32"), required=False, variadic=True
                ),
            },
            outputs={},
            pure=False,
            stateful=True,
            allowed_contexts={"actor", "learner"},
            side_effects=["logging"],
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="control"
        ),
    )

    # --- DQN Specific Specs ---
    # 1. QValuesSingle
    register_spec(
        "QValuesSingle",
        OperatorSpec.create(
            name="QValuesSingle",
            inputs={"obs": SingleObs},
            outputs={"q_values": SingleQ},
            pure=True,
            deterministic=True,
            allowed_contexts={"actor", "learner"},
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            parameter_handles=["model_handle"],
            domain_tags={"q_learning"},
            math_category="elementwise"
        ),
    )

    # 2. TDLoss
    register_spec(
        "TDLoss",
        OperatorSpec.create(
            name="TDLoss",
            inputs={"batch": TransitionBatch},
            outputs={"loss": Scalar("float32")},
            pure=True,
            deterministic=True,
            allowed_contexts={"actor", "learner"},
            differentiable=True,
            creates_grad=True,
            consumes_grad=False,
            updates_params=False,
            parameter_handles=["model_handle", "target_handle"],
            domain_tags={"q_learning"},
            math_category="loss"
        ),
    )

    # 2b. QForward
    register_spec(
        "QForward",
        OperatorSpec.create(
            name="QForward",
            inputs={"obs": BatchObs},
            outputs={"q_values": BatchQ},
            pure=True,
            deterministic=True,
            allowed_contexts={"actor", "learner"},
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            parameter_handles=["model_handle"],
            domain_tags={"q_learning"},
            math_category="elementwise"
        ),
    )
    # Legacy alias
    register_spec("QValuesBatch", get_spec("QForward"))

    # 2c. ReduceMean
    register_spec(
        "ReduceMean",
        OperatorSpec.create(
            name="ReduceMean",
            inputs={"input": BatchQ},
            outputs={"output": Scalar("float32")},
            pure=True,
            allowed_contexts={"actor", "learner"},
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="reduction"
        ),
    )

    # 2d. GetField (Generic)
    register_spec(
        "GetField",
        OperatorSpec.create(
            name="GetField",
            inputs={"input": TransitionBatch},  # Could be generic Schema
            outputs={"output": BatchObs},  # Could be generic Spec
            pure=True,
            allowed_contexts={"actor", "learner"},
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="elementwise"
        ),
    )

    # 3. Optimizer
    register_spec(
        "Optimizer",
        OperatorSpec.create(
            name="Optimizer",
            inputs={"loss": Scalar("float32")},
            outputs={"loss_val": Scalar("float32")},
            pure=False,
            stateful=True,
            allowed_contexts={"learner"},
            side_effects=["model_update"],
            differentiable=True,
            creates_grad=False,
            consumes_grad=True,
            updates_params=True,
            math_category="optimizer"
        ),
    )

    # 4. LinearDecay
    register_spec(
        "LinearDecay",
        OperatorSpec.create(
            name="LinearDecay",
            inputs={"clock": PortSpec(spec=Scalar("int64"), required=False)},
            outputs={"epsilon": Scalar("float32")},
            pure=True,
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="elementwise"
        ),
    )

    # 5. Exploration (Epsilon-Greedy)
    register_spec(
        "Exploration",
        OperatorSpec.create(
            name="Exploration",
            inputs={
                "q_values": SingleQ,
                "epsilon": PortSpec(spec=Scalar("float32"), required=False),
            },
            outputs={"action": Scalar("int64")},
            pure=False,
            deterministic=False,  # Stochastic due to epsilon
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            math_category="distribution"
        ),
    )

    register_spec(
        "GatherActionQ",
        OperatorSpec.create(
            name="GatherActionQ",
            inputs={"q_values": BatchQ, "actions": Scalar("int64")},
            outputs={"q_selected": Scalar("float32")},
            differentiable=True,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            domain_tags={"q_learning"},
            math_category="elementwise"
        ),
    )

    register_spec(
        "BellmanTarget",
        OperatorSpec.create(
            name="BellmanTarget",
            inputs={
                "next_q_values": BatchQ,
                "rewards": Scalar("float32"),
                "dones": Scalar("float32"),
            },
            outputs={"target": Scalar("float32")},
            differentiable=False,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            domain_tags={"q_learning"},
            math_category="elementwise"
        ),
    )
