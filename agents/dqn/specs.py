from runtime.specs import (
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
        "Source", OperatorSpec.create("Source", inputs={}, outputs={}, pure=True)
    )
    register_spec(
        "ReplayQuery",
        OperatorSpec.create(
            "ReplayQuery",
            inputs={},
            outputs={"default": TransitionBatch},
            pure=False,
            stateful=True,
            requires_buffers=["main"],
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
            side_effects=["target_update"],
            requires_models=["source", "target"],
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
            side_effects=["logging"],
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
        ),
    )

    # 2b. QValuesBatch
    register_spec(
        "QValuesBatch",
        OperatorSpec.create(
            name="QValuesBatch",
            inputs={"obs": BatchObs},
            outputs={"q_values": BatchQ},
            pure=True,
            deterministic=True,
        ),
    )

    # 2c. ReduceMean
    register_spec(
        "ReduceMean",
        OperatorSpec.create(
            name="ReduceMean",
            inputs={"input": BatchQ},
            outputs={"output": Scalar("float32")},
            pure=True,
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
            side_effects=["model_update"],
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
        ),
    )
