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
# Or with source port:
graph.add_edge("obs", "q_net", src_port="observation", dst_port="obs")
```

Returns:

```
Edge with explicit destination port for input dictionary key
```

Only if:

- src and dst exist in graph.nodes
- No cycle introduced (DAG constraint)

---

### Edge Types

Need:

```python
from core.graph import EdgeType
graph.add_edge("a", "b", edge_type=EdgeType.DATA)   # Data flow
graph.add_edge("a", "b", edge_type=EdgeType.CONTROL) # Control dependency
```

Edge types:

- `EdgeType.DATA`: Data flow (default)
- `EdgeType.CONTROL`: Control dependency (execution order only)
- `EdgeType.EFFECT`: Side effect

---

### Graph Serialization

Need:

```python
# Serialize to dict (JSON-compatible)
data = graph.to_dict()

# Deserialize from dict
restored = Graph.from_dict(data)
```

Returns:

```
to_dict: {"nodes": {...}, "edges": [...], "parameters": {...}}
from_dict: Graph with same structure
```

Also stores `graph.parameters` after compilation (parameter handle mapping).

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

## Graph Compilation

### Compile Graph

Need:

```python
from compiler.pipeline import compile_graph

compiled = compile_graph(
    graph,
    strict=False,
    model_handles={"online_q", "target_q"},
    buffer_handles={"main"},
    context="both",          # "actor", "learner", or "both"
    optimize=True,           # Run optimization passes
    autobatch=False,         # Vectorize for batching
    autodiff_lowering=True,  # Insert Backward/GradBuffer nodes
)
```

Compilation pipeline (in order):

1. **validate_metadata**: Check all node types have specs
2. **infer_shapes**: Propagate TensorSpec through graph
3. **autodiff** (if enabled): Insert Backward/GradBuffer nodes
4. **autobatch** (if enabled): Vectorize single-step to batched
5. **optimize_graph**: Dead node elimination, fusion
6. **validate_structure**: No cycles, reachable nodes
7. **validate_ports**: Schema compatibility at edges
8. **validate_rl_semantics**: On-policy/off-policy rules
9. **validate_handles**: Model/buffer handles exist
10. **validate_purity**: No side effects in wrong context
11. **validate_grad_semantics**: Gradient lifecycle safety
12. **validate_ir_purity**: No live objects in params
13. **collect_trainable_parameters**: Populate graph.parameters
14. **analyze_gradients**: Report gradient flow issues

Returns:

```
Graph with validated structure, parameters populated, optimizations applied
```

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

### Default Fusion Rules

The optimizer includes default fusion rules:

```python
# Pre-registered in OPTIMIZER_ENGINE
FusionRule(
    name="greedy_policy",
    pattern=["QValuesSingle", "Argmax"],
    replacement="GreedyPolicy"
)
```

Fuses Q-values computation + argmax into single greedy policy node.

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
- `config` - the DQNConfig object

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

### Port Errors (E)

- E203: Typo in dst_port - provides "Did you mean..." suggestion
- E204: Port type mismatch - shows Expected/Got with path
- E205: Required port missing
- E206: Ambiguous autowire - multiple compatible ports

### Schema Errors (E)

- E310: Missing field in schema
- E311: Field type/dtype mismatch

### Gradient Errors (G)

- G001: Optimizer without preceding Backward
- G002: Backward without optimizer
- G003: Same parameters updated twice in one step
- G004: Parameter update in actor (inference) context
- G005: Gradient nodes in actor graph

---

## Optimization Report Details

The OptimizationReport tracks multiple optimization categories:

```python
report = OptimizationReport()
optimized = optimize_graph(graph, report=report)
print(report)
```

Output sections:

- **Detected trainable params**: Lists parameter handles used in graph
- **Inserted backward pass**: Shows loss → Backward(node) mappings
- **Dead Node Elimination**: Number and list of removed nodes
- **Applied rule**: Fusion rule details (pattern → replacement)
- **Skipped fusion**: Reasons fusion was skipped (e.g., "backward boundary blocks fusion")
- **Applied no_grad hoist**: Target network branches wrapped in no_grad

---

## Port Features

### Optional Ports

Ports can be marked optional in OperatorSpec:

```python
OperatorSpec.create(
    name="MyOp",
    inputs={
        "required": PortSpec(spec=SingleObs, required=True),
        "optional": PortSpec(spec=SingleObs, required=False),
    }
)
```

- Missing optional port: No error, uses default value or skips
- Missing required port: E205 error

### Default Value Injection

Missing optional ports can have default values:

```python
PortSpec(spec=SingleObs, required=False, default=0.5)
```

Executor automatically injects defaults when port is missing.

### Autowiring

When dst_port is not specified, the system auto-wires to the first compatible port:

- One compatible port: Auto-connect (passes validation)
- Multiple compatible ports: E206 ambiguous error

### Error Suggestions

The validator provides helpful suggestions:

- E203: "Did you mean dst_port='correct_port'?"
- E204: "Suggestion: Use dst_port='single_obs'"

---

## Cost Model

### Operator Cost Estimates

OperatorSpec can include:

- `estimated_flops`: FLOPs for this operation
- `memory_reads`: Bytes read from memory
- `kernel_launch_cost`: Overhead in microseconds

Used by optimizer for scheduling decisions.

---

## Memory Optimizations

### Activation Checkpointing

Need:

```python
from compiler.passes.memory_optimizations import apply_activation_checkpointing
graph = apply_activation_checkpointing(graph)
```

Reduces memory by recomputing activations during backward pass instead of storing them.

---

## Gradient Analysis

### Analyze Gradients

Need:

```python
from compiler.passes.analyze_gradients import analyze_gradients
report = analyze_gradients(graph)
```

Returns GradientReport with:

- `params_with_grad`: Parameters in gradient path
- `params_without_grad`: Parameters not receiving gradients
- `unused_branches`: Differentiable nodes with no consumers
- `warnings`: List of issues found

Detects:

- Detached gradient flow (non-differentiable operators in path)
- Dead parameters (no path to loss)
- Unused differentiable branches

---

## Autodiff Lowering

### Insert Backward Nodes

Need:

```python
compiled = compile_graph(graph, context="learner", autodiff_lowering=True)
```

Automatically inserts:

- `Backward` node downstream of loss nodes
- `GradBuffer` node for gradient storage
- Connects Loss → Backward → Optimizer

---

## Collect Trainable Parameters

### Parameter Handle Mapping

Need:

```python
from compiler.passes.collect_trainable_parameters import collect_trainable_parameters
param_map = collect_trainable_parameters(graph)
```

Returns:

```python
{
    "online_q": ["q_values", "loss"],
    "target_q": ["target_sync"]
}
```

Maps parameter handles to lists of node IDs that use them.

---

## Gradient Validation

### Validate Gradient Semantics

Need:

```python
from compiler.passes.validate_grad_semantics import validate_grad_semantics
report = validate_grad_semantics(graph, context="learner")
```

Validates gradient lifecycle:

- G001: Optimizer without preceding Backward
- G002: Backward without optimizer
- G003: Same params updated twice in one step
- G004: Inference updates params (actor context)
- G005: Actor graph contains gradient nodes

---

## Optimization Report

### Track Optimizations

Need:

```python
from compiler.optimizer import optimize_graph, OptimizationReport
report = OptimizationReport()
optimized = optimize_graph(graph, report=report)
```

Report contains:

- `steps`: List of optimization steps applied
- `dead_nodes_removed`: Nodes eliminated
- `fusion_count`: Number of fusions performed

Each step records:

- `rule_name`: Name of rule applied
- `pattern`: Original node chain
- `replacement`: Fused node type
- `removed_nodes`: Nodes eliminated

---

## Explainable Errors

### Port Mismatch (E204)

Detailed error showing:

```
q_net.obs <- sampler
Expected: SingleObs[float32, shape=(4,)]
Got:      TransitionBatch[obs: float32, action: int64, ...]
```

### Field Mismatch (E311)

Shows exact field causing incompatibility:

```
loss.batch <- sampler
Field 'action' mismatch:
Expected: int64
Got:      float32
```

### Missing Field (E310)

Shows missing required field:

```
loss.batch <- sampler
Field 'reward' missing from schema
Expected in batch: reward: float32
```

---

## Graph Parameters

### Trainable Parameters in Graph

Graph stores parameter metadata:

```python
graph.parameters = {
    "online_q": ["q_values", "loss"],
    "target_q": ["target_sync"]
}
```

Set during compilation via `collect_trainable_parameters`.

---

## Validation Codes (Extended)

### Gradient (G)
| Code | Description |
|------|-------------|
| G001 | Optimizer without Backward |
| G002 | Backward without Optimizer |
| G003 | Parameter updated twice |
| G004 | Inference updates params |
| G005 | Gradient nodes in actor |

### Explainability (E)
| Code | Description |
|------|-------------|
| E204 | Port mismatch with path |
| E310 | Missing field in schema |
| E311 | Field type/dtype mismatch |