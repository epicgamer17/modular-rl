# ADR-0024: Metrics as Blackboard Outputs

## Status
Accepted

## Context
In many Reinforcement Learning (RL) codebases, logging is implemented as "scattered side effects." Code for computing a policy loss is often intertwined with code that sends that loss to a specific logger (e.g., `wandb.log({"loss": loss.item()})`).

This scattering causes several architectural problems:
- **Tight Coupling**: Mathematical components become dependent on specific logging libraries.
- **Inflexible Monitoring**: It is difficult to change *where* or *how often* a metric is logged without modifying the core logic.
- **Observability Gaps**: Silent failures in custom logging code are hard to detect.
- **Testing Friction**: Unit testing a component requires mocking global logging collectors.

## Options Considered
### Option 1: Inline Logging (Standard)
- **Pros**
    - Immediate and familiar.
- **Cons**
    - Violates the **Pure Dataflow Units (ADR-0017)** principle.
    - Hard to aggregate or suppress metrics globally.

### Option 2: Metrics as Blackboard Outputs (Chosen)
- **Pros**
    - **Architecture Cleanliness**: Components only care about math; they write their results (including diagnostic ones) to the Blackboard and exit.
    - **Centralized Routing**: Specialized **Metric Sink Components** (see ADR-0018) can be plugged in to collect, aggregate, and export metrics.
    - **Uniform Validation**: Metrics follow the same **Semantic Typing (ADR-0016)** and **Validation (ADR-0015)** as core training data.
- **Cons**
    - Requires defining intermediate keys for every value worth logging.

## Decision
We will implement all diagnostic values and performance metrics must be treated as standard **Blackboard Outputs**.

Instead of performing a side-effect log inside a loss calculation:
```python
# FORBIDDEN
def compute_loss(self, ...):
    loss = ...
    wandb.log({"policy_loss": loss.item()})
    return loss
```

Components must declare and write to metric keys:
```python
# PREFERRED
# Contract: outputs = {"metrics.policy_loss": Metric[Scalar]}
def compute_loss(self, blackboard, ...):
    loss = ...
    blackboard.write("metrics.policy_loss", loss.detach())
```

Dedicated **Sink Components** (e.g., `WandBSink`, `ConsoleLogger`) are then placed at the end of the execution graph to consume these keys and perform the actual external side effect.

## Consequences
### Positive
- **Plug-and-Play Diagnostics**: You can swap from TensorBoard to W&B by simply changing one node in the execution graph, without touching any mathematical code.
- **Diagnostic Transparency**: The system's execution graph (ADR-0024) now shows exactly which metrics are being generated and where they are going.
- **Aggregated Logging**: Sinks can intelligently aggregate metrics (e.g., averaging over a window) before sending them over the network, improving performance.

### Negative / Tradeoffs
- **Key Proliferation**: The number of keys in the system will increase as every logged scalar needs a unique name and semantic type.

## Notes
This ADR formalizes metrics as first-class citizens of the dataflow, ensuring that the **Blind Learner** remains truly blind to the diagnostic infrastructure.