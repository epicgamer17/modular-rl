from ops.control.loops import op_loop, op_minibatch_iterator
from runtime.registry import OperatorSpec, PortSpec

SOURCE_SPEC = OperatorSpec.create(
    "Source",
    inputs={},
    outputs={},
    pure=True,
    allowed_contexts={"actor", "learner"},
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)

GET_FIELD_SPEC = OperatorSpec.create(
    name="GetField",
    inputs={"input": PortSpec(spec=None)},
    outputs={"output": PortSpec(spec=None)},
    pure=True,
    math_category="elementwise",
    allowed_contexts={"actor", "learner"},
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)
