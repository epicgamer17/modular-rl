# Capabilities

This document catalogs the capabilities of the RL IR Semantic Kernel system, organized by functional area. Each capability defines the interface, preconditions, and expected behavior.

---

## Graph Construction

### Create Empty Graph

Need:

```python
graph = Graph()
```

Returns:

```
Graph with empty nodes dict and edges list
```

Only if:

- No preconditions

---

### Add Node to Graph

Need:

```python
graph.add_node(
    node_id="q_values",
    node_type="QValuesSingle",
    params={"model_handle": "online_q"},
    tags=["actor"]
)
```

Returns:

```
Graph with node added to nodes dict
```

Only if:

- node_id is unique in graph
- node_type is valid string (registered operator or will error at runtime)

---

### Add Edge with Named Port

Need:

```python
graph.add_edge("sampler", "loss", dst_port="batch")
graph.add_edge("q_values", "metrics", dst_port="avg_q")
```

Returns:

```
Edge with explicit destination port for input dictionary key
```

Only if:

- src and dst exist in graph.nodes
- No cycle introduced (DAG constraint)

---

### Get Topological Sort

Need:

```python
# Internal in executor - Kahn's algorithm
order = _topological_sort(graph)
```

Returns:

```
[List[NodeId]] - nodes in execution order
```

Only if:

- Graph is a valid DAG (no cycles)

---

## Schema & Type System

### Create Schema

Need:

```python
schema = Schema(fields=[Field(name="obs", spec=TensorSpec(...))])
```

Returns:

```
Schema with field list
```

Only if:

- Field names are unique

---

### Create TensorSpec

Need:

```python
spec = TensorSpec(shape=(4, 84, 84), dtype=torch.float32, tags=[])
```

Returns:

```
TensorSpec with shape and dtype
```

Only if:

- shape is tuple of ints
- dtype is valid torch dtype

---

## Execution

### Execute Graph

Need:

```python
results = execute(graph, initial_inputs, context)
```

Returns:

```
Dict[NodeId, RuntimeValue] - node outputs wrapped in RuntimeValue
```

Only if:

- Graph has valid topological order
- All node types have registered operators

---

### Register Operator

Need:

```python
def my_op(node, inputs, context):
    return Value(output_data)

register_operator("MyNodeType", my_op)
```

Returns:

```
Operator available in OPERATOR_REGISTRY
```

Only if:

- Function signature: (Node, Dict, ExecutionContext) -> Any

---

## Runtime Values

Replaces None-based control flow with explicit tagged wrappers.

### Value (Success with Data)

Need:

```python
Value(tensor)
```

Returns:

```
RuntimeValue with has_data=True
```

---

### NoOp (Success, No Data)

Need:

```python
NoOp()
```

Returns:

```
RuntimeValue with has_data=False
```

---

### Skipped (Intentional Skip)

Need:

```python
Skipped("buffer_size_42_under_min_64")
```

Returns:

```
RuntimeValue propagating skip with reason
```

---

### MissingInput (Required Input Missing)

Need:

```python
MissingInput("batch")
```

Returns:

```
RuntimeValue indicating required input was not provided
```

---

### Auto-Skip Propagation

The executor automatically skips nodes if any input is Skipped or MissingInput, propagating the status downstream without requiring operators to check manually.

---

## Execution Context

### Create Execution Context

Need:

```python
context = ExecutionContext(
    seed=42,
    shard_id=0,
    model_registry=ModelRegistry(),
    buffer_registry=BufferRegistry()
)
```

Counters (all start at 0):

- `actor_step` - rollout steps
- `env_step` - environment interactions
- `learner_step` - training updates
- `sync_step` - target network syncs
- `episode_step` - steps in current episode
- `episode_count` - total episodes
- `global_step` - total steps

RNG: `context.rng` (random.Random), seeded with `seed + shard_id * 1000000`

---

### Derive Child Context

Need:

```python
child = context.derive(step_id=context.step_id + 1)
```

Returns:

```
New context with inherited state, incremented step_id
```

---

### Get Model by Handle

Need:

```python
model = context.get_model("online_q")
```

Returns:

```
Live nn.Module from ModelRegistry
```

---

### Get Buffer by Handle

Need:

```python
buffer = context.get_buffer("main")
```

Returns:

```
Live ReplayBuffer from BufferRegistry
```

---

## Actor Runtime

### Create Actor Runtime

Need:

```python
actor = ActorRuntime(
    interact_graph=graph,
    env=gym.make("CartPole-v1"),
    replay_buffer=buffer,
    recording_fn=None
)
```

---

### Step

Need:

```python
trace = actor.step(context)
```

Returns:

```
Dict with obs, action, reward, next_obs, done, metadata
```

- Increments `context.actor_step`, `context.env_step`, `context.episode_step`
- Creates ActorSnapshot for nodes with param_store
- Calls recording_fn or adds to replay_buffer

---

## Learner Runtime

### Create Learner Runtime

Need:

```python
learner = LearnerRuntime(train_graph=graph, replay_buffer=buffer)
```

---

### Update Step

Need:

```python
results = learner.update_step(batch=None, context=context)
```

If batch=None, samples from internal replay_buffer.

Returns:

```
Dict of node outputs (unwrapped from RuntimeValue)
```

- Increments `context.learner_step`

---

## State Management

### Replay Buffer

Need:

```python
buffer = ReplayBuffer(capacity=10000)
buffer.add(transition_dict)
batch = buffer.sample(batch_size=32, seed=42)
```

Returns:

```
List of transition dicts
```

Thread-safe with internal lock.

---

### Sample Query

Need:

```python
batch = buffer.sample_query(
    batch_size=32,
    filters={"policy_version": 2},
    temporal_window=1000,
    contiguous=False,
    seed=42
)
```

Returns:

```
Filtered, ordered batch
```

---

### Parameter Store

Need:

```python
state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
store = ParameterStore(state)
store.update_state(new_state)
version = store.version  # increments on update
```

Returns:

```
Versioned parameter storage
```

---

### Optimizer State

Need:

```python
opt_state = OptimizerState(optimizer, grad_clip=0.5)
stats = opt_state.step(loss)
```

Guarantees (in order):

1. `zero_grad(set_to_none=True)`
2. `loss.backward()`
3. `clip_grad_norm_` (if grad_clip set)
4. `optimizer.step()`

Returns:

```
{"loss": float, "grad_norm": float, "lr": float}
```

---

### Model Registry

Need:

```python
registry = ModelRegistry()
registry.register("online_q", q_net)
model = registry.get("online_q")
```

---

### Buffer Registry

Need:

```python
registry = BufferRegistry()
registry.register("main", buffer)
buf = registry.get("main")
```

---

## Exploration & Scheduling

### Epsilon-Greedy (Deterministic)

Need:

```python
# In exploration.py operator
epsilon = inputs.get("epsilon", 1.0)
if context.rng.random() < epsilon:
    action = context.rng.randint(0, num_actions - 1)
else:
    action = q_values.argmax(dim=-1)
```

Returns:

```
Action using context.rng (not global random)
```

Only if:

- Uses context.rng for determinism
- Shard-aware: seed = base_seed + shard_id * 1000000

---

### Linear Decay

Need:

```python
# In schedule.py operator
# Parameters: start_value, end_value, total_steps, clock (default: env_step)
step = inputs.get("clock", context.env_step)
progress = min(step / total_steps, 1.0)
value = start_value + (end_value - start_value) * progress
```

Returns:

```
Linearly decayed value
```

---

## Target Network Sync

### Target Sync Operator

Need:

```python
# Parameters: source_handle, target_handle, tau, sync_frequency, sync_on
if context.learner_step % sync_frequency == 0:
    if tau == 1.0:
        target_params = source_params.clone()
    else:
        target_params = tau * source_params + (1 - tau) * target_params
    return Value(updated=True)
return NoOp()
```

Returns:

```
Hard/soft sync when frequency condition met
```

---

## Metrics

### Metrics Sink

Need:

```python
# Node parameters: buffer_id, log_frequency
# Expected input ports: loss, avg_q, batch, epsilon
# Tracked: actor_step, learner_step, episode_count, sync_step, SPS
```

Returns:

```
Logs to stdout every N steps, returns metrics dict
```

---

## Collator

### Replay Collator

Need:

```python
collator = ReplayCollator(schema)
batch = collator.collate(transitions)
```

Schema:

```python
replay_schema = Schema(fields=[
    Field(name="obs", spec=TensorSpec(shape=(4, 84, 84), dtype=torch.float32)),
    Field(name="action", spec=TensorSpec(shape=(1,), dtype=torch.long)),
    Field(name="reward", spec=TensorSpec(shape=(1,), dtype=torch.float32)),
    Field(name="done", spec=TensorSpec(shape=(1,), dtype=torch.float32)),
])
```

Returns:

```
{
    "obs": torch.Tensor [batch_size, ...],
    "action": torch.Tensor [batch_size] (long),
    "reward": torch.Tensor [batch_size] (float32),
    "done": torch.Tensor [batch_size] (float32),
    "metadata": list of dicts
}
```

---

## Operators (Node Types)

### QValuesSingle

- Purpose: Acting (single observation)
- Input: `obs` tensor [obs_dim]
- Output: Q-values [num_actions]
- Uses: ActorSnapshot for functional_call

---

### QValuesBatch

- Purpose: Training (batched observations)
- Input: `batch` dict with "obs" key
- Output: Q-values [batch_size, num_actions]
- Uses: Current parameters

---

### ReplayQuery

- Purpose: Sample from replay buffer
- Parameters: `buffer_id`, `batch_size`, `min_size`, `filters`, `temporal_window`, `contiguous`, `collator`
- Returns: Batch or Skipped if min_size not met

---

### TargetSync

- Purpose: Sync target network parameters
- Parameters: `source_handle`, `target_handle`, `tau`, `sync_frequency`, `sync_on`

---

### MetricsSink

- Purpose: Collect and log metrics
- Parameters: `buffer_id`, `log_frequency`
- Input ports: `loss`, `avg_q`, `batch`, `epsilon`

---

### LinearDecay

- Purpose: Hyperparameter schedule
- Parameters: `start_value`, `end_value`, `total_steps`
- Input: `clock` (optional, defaults to context.env_step)

---

### EpsilonGreedy (Exploration)

- Purpose: Action selection with exploration
- Parameters: `num_actions`
- Input ports: `q_values`, `epsilon`

---

## Transfer Operators

### Transfer to Device

Need:

```python
op_transfer_to_device(ref, device="cuda", context)
```

Returns:

```
DataRef on target device
```

---

### Transfer to CPU

Need:

```python
op_transfer_to_cpu(ref, context)
```

---

### Serialize

Need:

```python
bytes_data = op_serialize(ref, context)
```

---

### Deserialize

Need:

```python
ref = op_deserialize(bytes_data, context)
```

---

## Examples

### DQN Pipeline

```
interact_graph:
  obs_in (Source) -> QValuesSingle -> EpsilonGreedy -> action

train_graph:
  ReplayQuery -> QValuesBatch -> TDLoss -> Optimizer -> TargetSync -> MetricsSink
```

### PPO Pipeline

```
interact_graph:
  obs_in -> Policy -> Action -> env -> ...

train_graph:
  ReplayQuery (contiguous) -> GAE -> PPOObjective -> Optimizer
```

### NFSP Pipeline

```
Two buffers: RL and SL
MixtureActor for anticipatory sampling
Dual training graphs
```

### DAgger Pipeline

```
Two actors in same graph: Student + Expert
Expert labels captured in metadata
Custom recording function aggregates
```

---

## Operator Specifications

### Register Operator Spec

Need:

```python
spec = OperatorSpec.create(
    name="MyOp",
    inputs={"batch": TransitionBatch},
    outputs={"loss": ScalarLoss},
    pure=False,
    stateful=True,
    deterministic=False,
    requires_models=["main"],
    requires_optimizer=True,
    allowed_contexts={"learner"},
    differentiable=True,
    creates_grad=True,
    consumes_grad=True,
    updates_params=True,
    parameter_handles=["model_handle"],
    tags={"off_policy", "heavy"},
    estimated_flops=1000,
)
register_spec("MyOp", spec)
```

Spec properties:

- `pure`: No side effects
- `stateful`: Maintains internal state
- `deterministic`: Same inputs → same outputs
- `requires_models`: List of model handles needed
- `requires_optimizer`: Needs optimizer state
- `allowed_contexts`: {"actor", "learner"}
- `differentiable`: Can compute gradients
- `creates_grad`: Produces gradient tensors
- `consumes_grad`: Input requires gradients
- `updates_params`: Modifies model parameters
- `parameter_handles`: Which params are touched

---

### Get Operator Spec

Need:

```python
spec = get_spec("QValuesSingle")
```

Returns:

```
OperatorSpec or None if not registered
```

---

### Check Spec Compatibility

Need:

```python
is_compatible(source_spec, dest_spec)
```

Returns:

```
True if shapes, dtypes, and RL types match
```

---

## Rewrite & Fusion

### RewriteEngine

Need:

```python
engine = RewriteEngine()
```

Methods:

- `add_rule(FusionRule(...))`
- `apply(graph)` - returns optimized graph

---

### FusionRule

Need:

```python
rule = FusionRule(
    name="fuse_a_b",
    pattern=["OpA", "OpB"],
    replacement="FusedOp"
)
```

Fuses chain A → B into single FusedOp node.

---

### Find Linear Chain

Need:

```python
find_linear_chain(graph, ["OpA", "OpB", "OpC"])
```

Returns:

```
List of NodeIds forming exact chain
```

Only if:

- Exact edge chain exists
- Single consumer per node
- No branching

---

## Autobatching

### Vectorize Graph

Need:

```python
from compiler.passes.autobatch import vectorize_graph
batch_graph = vectorize_graph(graph)
```

Transforms single-step schemas to batched:

- [D] → [B, D]
- Adds "batched" tag to specs and RLTypes
- Nodes with "no_batch" tag are skipped

---

## Parameter Conventions

### Canonical Parameter Names

Required:

- `model_handle` - reference to model (not `q_net`, `policy`)
- `target_handle` - reference to target network
- `optimizer_handle` - reference to optimizer state
- `buffer_id` - reference to replay buffer

Banned:

- `source_handle`, `q_handle`, `opt_state`, `q_net`, `target_q`, `replay_buffer`

---

## Type System

### RLType Categories

```python
RLTypeCategory.TENSOR
RLTypeCategory.TRAJECTORY
RLTypeCategory.EPISODE
RLTypeCategory.DISTRIBUTION
RLTypeCategory.POLICY_SNAPSHOT
RLTypeCategory.REPLAY_BATCH
RLTypeCategory.SCALAR_METRIC
RLTypeCategory.RNG_KEY
RLTypeCategory.HIDDEN_STATE
```

### TensorType

```python
TensorType(shape=("B", 4), dtype="float32", tags={"obs", "batched"})
```

### DistributionType

```python
DistributionType(dist_type="Categorical", is_logits=True)
```

### PolicySnapshotType

```python
PolicySnapshotType(version=5)
```

---

## Graph Serialization

### Roundtrip Test

Graphs can be serialized and deserialized while preserving structure.

---

## Autowiring

### Auto-wire Node Connections

System can automatically infer and create edges between nodes based on port specifications.

---

## Agent System

### DQNAgent

Need:

```python
from agents.dqn.agent import DQNAgent
from agents.dqn.config import DQNConfig

config = DQNConfig(
    obs_dim=4,
    act_dim=2,
    hidden_dim=64,
    lr=1e-3,
    buffer_capacity=1000
)
agent = DQNAgent(config)
```

Creates:

- QNetwork + TargetNetwork
- OptimizerState
- ReplayBuffer
- ModelRegistry, BufferRegistry, OptimizerRegistry
- Actor graph + Learner graph
- ReplayCollator with schema

Methods:

- `get_execution_context(seed)` - creates context with registries
- `compile(strict=False)` - validates both graphs
- `actor_graph`, `learner_graph` - the IR graphs

---

### PPOAgent

Similar structure with:

- On-policy buffer (flush after each epoch)
- GAE computation
- PPO objective with clipping

---

## Optimizer Registry

Need:

```python
from runtime.state import OptimizerRegistry

registry = OptimizerRegistry()
registry.register("main_opt", opt_state)
opt = registry.get("main_opt")
```

---

## Validation Codes

### Purity Errors

- P001: nn.Module in params
- P002: Optimizer in params
- P003: ReplayBuffer in params
- P004: Callable/lambda in params

### Metadata Errors

- M001: Unregistered operator type

### Structure Errors

- S001: Cycle detected
- S002: Disconnected node

---

## Cost Model

### Operator Cost Estimates

OperatorSpec can include:

- `estimated_flops`: FLOPs for this operation
- `memory_reads`: Bytes read from memory
- `kernel_launch_cost`: Overhead in microseconds

Used by optimizer for scheduling decisions.