from runtime.specs import (
    OperatorSpec,
    PortSpec,
    register_spec,
    get_spec,
    Tensor,
    Scalar,
    SingleObs,
    BatchObs,
    Distribution,
)


def register_ppo_specs():
    """Registers specs for PPO operators."""

    register_spec(
        "PolicyForward",
        OperatorSpec.create(
            name="PolicyForward",
            inputs={"obs": SingleObs},
            outputs={
                "action": Scalar("int64"),
                "log_prob": Scalar("float32"),
                "values": Scalar("float32"),
                "policy_version": Scalar("int64"),
            },
            pure=True,
            deterministic=False,
            allowed_contexts={"actor", "learner"},
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            parameter_handles=["model_handle"],
        ),
    )

    # TODO: remove this legacy alias
    # Legacy alias
    register_spec("PPO_PolicyActor", get_spec("PolicyForward"))

    register_spec(
        "PPO_Objective",
        OperatorSpec.create(
            name="PPO_Objective",
            inputs={"batch": PortSpec(spec=None, required=True)},  # Schema-less for now
            outputs={"loss": Scalar("float32")},
            pure=True,
            deterministic=True,
            allowed_contexts={"learner"},
            differentiable=True,
            creates_grad=True,
            consumes_grad=False,
            updates_params=False,
            parameter_handles=["model_handle"],
            domain_tags={"policy_gradient"}
        ),
    )

    register_spec(
        "PPO_Optimizer",
        OperatorSpec.create(
            name="PPO_Optimizer",
            inputs={"loss": Scalar("float32")},
            outputs={"loss_val": Scalar("float32")},
            pure=False,
            stateful=True,
            allowed_contexts={"learner"},
            differentiable=True,
            creates_grad=False,
            consumes_grad=True,
            updates_params=True,
            parameter_handles=["model_handle"],
        ),
    )

    register_spec(
        "Ratio",
        OperatorSpec.create(
            name="Ratio",
            inputs={
                "new_log_probs": Scalar("float32"),
                "old_log_probs": Scalar("float32"),
            },
            outputs={"ratio": Scalar("float32")},
            differentiable=True,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            domain_tags={"policy_gradient"}
        ),
    )

    register_spec(
        "Clip",
        OperatorSpec.create(
            name="Clip",
            inputs={"input": Scalar("float32")},
            outputs={"output": Scalar("float32")},
            differentiable=True,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            domain_tags={"policy_gradient"}
        ),
    )

    register_spec(
        "SurrogateMin",
        OperatorSpec.create(
            name="SurrogateMin",
            inputs={
                "ratio": Scalar("float32"),
                "clipped_ratio": Scalar("float32"),
                "advantages": Scalar("float32"),
            },
            outputs={"surrogate_loss": Scalar("float32")},
            differentiable=True,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            domain_tags={"policy_gradient"}
        ),
    )

    register_spec(
        "ValueLoss",
        OperatorSpec.create(
            name="ValueLoss",
            inputs={
                "values": Scalar("float32"),
                "returns": Scalar("float32"),
                "old_values": PortSpec(spec=Scalar("float32"), required=False),
            },
            outputs={"value_loss": Scalar("float32")},
            differentiable=True,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )

    register_spec(
        "Entropy",
        OperatorSpec.create(
            name="Entropy",
            inputs={"probs": PortSpec(spec=None)},
            outputs={"entropy": Scalar("float32")},
            differentiable=True,
            allowed_contexts={"learner"},
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            domain_tags={"policy_gradient"}
        ),
    )
