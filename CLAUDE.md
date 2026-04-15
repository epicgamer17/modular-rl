# CLAUDE.md — Project Rules & Standards

## Quick Start

```bash
# Install main package and environments package (both required)
pip install -e .
pip install -e custom_gym_envs_pkg/
pip install einops  # required but absent from requirements.txt
```

## Testing

```bash
pytest tests/ -m unit                  # fast, isolated logic checks
pytest tests/ -m integration           # component interaction tests
pytest tests/ -m slow                  # long training loops
pytest tests/ -m regression            # historical repro checks

pytest tests/ -p no:cov -m unit        # skip coverage for faster iteration
```

Coverage threshold is **80%** (enforced by `pytest.ini` via `--cov-fail-under=80`).

---

## Core Philosophy

This framework is built on **Strict Separation of Concerns and Perfect Polymorphism**. Deep RL systems collapse when neural network math bleeds into game logic, or when optimization math bleeds into tree search logic.

- **The Blind Learner:** `BlackboardEngine` must be completely blind to the network's architecture. It calls `learner_inference(batch)`, receives raw math, and routes it to the `LossAggregator`.
- **The Blind Actor/Tree:** MCTS and Actor treat `network_state` (a generic recurrent state dictionary) as an **Opaque Token** — they store and pass it back without inspecting it.
- **Game Logic Isolation:** The Neural Network is a pure math function. Action masking and legal move filtering happen strictly *outside* the network (in `BaseActionSelector` subclasses or the Search Tree).

---

## Directory Structure & Domain Boundaries

| Directory | Domain | Rule |
|---|---|---|
| `core/` | Primitives | The DAG Execution Engine. Contains `Blackboard`, `BlackboardEngine`, `PipelineComponent`, `ExecutionGraph`, contract system (`Key`, `ShapeContract`, `SemanticType`), and debugging tools (`blackboard_diff`). |
| `modules/` | Pure PyTorch Neural Networks | No knowledge of envs, buffers, or loss functions. Tensors in → tensors out. |
| `agents/learners/` | Loss Pipeline & Optimizer | Calls `learner_inference(batch) -> LearningOutput`, unpacks it, routes raw tensors to `LossAggregator`. Never contains graph routing loops. |
| `agents/action_selectors/` | Math → Action bridge | Where game rules meet network math. Action masking happens here via `mask_actions()`. |
| `agents/trainers/` | Training orchestration | Instantiates and wires learner, workers, buffer, and executors. |
| `agents/workers/` | Actor & testing workers | `BaseActor` subclasses (`actors.py`, `puffer_actor.py`) run env loops and write to the replay buffer. |
| `agents/executors/` | Process management | `local` and `torch_mp` backends for scaling workers. |
| `replay_buffers/` | High-performance data storage | Stores immutable facts (uint8). Mathematical targets computed on-the-fly in `sample()` via `OutputProcessor`. |
| `losses/` | Loss computation | `LossModule` subclasses + `LossAggregator`. Take raw tensors — never Distribution objects. |
| `search/` | CPU-bound MCTS | Lives entirely on CPU. Interacts with GPU only by batching opaque states. |
| `custom_gym_envs_pkg/custom_gym_envs/` | All RL Environments | **All environments must be in this package.** |
| `configs/` | Configuration | YAML-backed, composable via mixins (`OptimizationConfig`, `ReplayConfig`, `SearchConfig`, etc.) |
| `offline_data/` | Offline/human replay data | Stored separately; not committed to git. |

### The 3 Stages of Preprocessing
1. **Game Preprocessing** (Env Wrappers in `utils/wrappers.py`): Resize, grayscale, frame-stack → raw NumPy `uint8`.
2. **Tensor Routing** (Action Selectors & Buffer Samplers): NumPy → Tensor, `unsqueeze(0)`, move to device. No dtype change.
3. **Neural Preprocessing** (Network Backbones): On-GPU. Cast `uint8` → `float32` etc.

---

## Core Class Contracts

### `ModularAgentNetwork` — The 2 Public Inference APIs

`ModularAgentNetwork` (in `modules/agent_nets/modular.py`) is the switchboard between RL system and PyTorch sub-modules. It dynamically initialises components based on config type (PPO, Rainbow, Supervised).

- **`obs_inference(obs) -> InferenceOutput`** — Used by Actor/MCTS for real-world root states. Returns semantic objects (expected values, `Distribution` objects). **Never raw logits.**
- **`learner_inference(batch) -> Dict[str, Tensor]`** — Used by the Learner for batched/sequential inference. Returns purely mathematical objects (raw logits, C51 atoms) for stable cross-entropy loss.
- **`get_learner_contract() -> Dict[str, Type[SemanticType]]`** — Exposes the semantic output contract of the network for automated DAG discovery.

For MuZero latent stepping, use the `MuzeroWorldModel` directly:
- **`world_model.recurrent_inference(state, action) -> WorldModelOutput`** — MCTS latent stepping.
- **`world_model.initial_inference(obs) -> WorldModelOutput`** — Initial features from observation.
- **`world_model.unroll_physics(initial_latent, actions, ...) -> PhysicsOutput`** — Unroll T steps for learner.

Must pack/unpack all sub-module RNN states into the `network_state` field (the Opaque Token).

### Key Output Types (all NamedTuples)

```python
# modules/world_models/inference_output.py

class InferenceOutput(NamedTuple):
    network_state: Any = None          # Opaque token (Recurrent state dictionary)
    value: float | Tensor = 0.0
    q_values: Optional[Tensor] = None
    policy: Optional[Distribution] = None
    reward: Optional[float | Tensor] = None
    chance: Optional[Distribution] = None
    to_play: Optional[int | Tensor] = None
    extras: Optional[dict] = None

class LearningOutput(NamedTuple):
    values: Optional[Tensor] = None
    policies: Optional[Tensor] = None  # raw logits
    q_values: Optional[Tensor] = None
    q_logits: Optional[Tensor] = None  # C51 atoms
    rewards: Optional[Tensor] = None
    to_plays: Optional[Tensor] = None
    latents: Optional[Tensor] = None
    latents_afterstates: Optional[Tensor] = None
    chance_logits: Optional[Tensor] = None
    chance_values: Optional[Tensor] = None
    target_latents: Optional[Tensor] = None
    chance_encoder_embeddings: Optional[Tensor] = None

class WorldModelOutput(NamedTuple):
    features: Tensor
    reward: Optional[Tensor] = None
    to_play: Optional[Tensor] = None
    q_values: Optional[Tensor] = None
    head_state: Any = None
    instant_reward: Optional[Tensor] = None
    afterstate_features: Optional[Tensor] = None
    chance: Optional[Tensor] = None

    latents: Tensor
    rewards: Tensor
    to_plays: Tensor
    latents_afterstates: Optional[Tensor] = None
    chance_logits: Optional[Tensor] = None
    # ... afterstate and chance encoder fields
```

### `BaseActionSelector`
```python
# agents/action_selectors/selectors.py
def select_action(agent_network, obs, info=None, network_output=None, exploration=None, **kwargs) -> Tuple[Tensor, Dict]
def mask_actions(values, legal_moves, mask_value=-inf, device=None) -> Tensor
```
Concrete implementations: `CategoricalSelector`, `EpsilonGreedySelector`, `ArgmaxSelector`, `NFSPSelector`.

Must call `obs_inference`. Must apply action mask directly to Distribution logits before sampling. After Softmax/Gumbel, re-apply mask to set illegal probability to exactly `0.0`.

### `BlackboardEngine`
```python
# core/blackboard_engine.py
BlackboardEngine(
    components: List[PipelineComponent],
    device: torch.device,
    initial_keys: Set[Key] = set(),
    strict: bool = False,      # Runtime validation via component.validate()
    lazy: bool = False,         # Demand-driven execution (skip satisfied components)
    diff: bool = False,         # Per-component blackboard change tracking
    **kwargs,                   # Forward-compatibility
)
def step(self, batch_iterator) -> Iterator[Dict[str, Any]]
```

#### DAG Build-Time Validation
The engine enforces contract consistency before the first training step via `validate_recipe()`:
1. **Dependency Resolution**: All `requires` paths must exist in the namespace.
2. **Semantic Compatibility**: `SemanticType` subclassing check (Provider ⊆ Consumer).
3. **Representation Consistency**: Exact match of metadata (vmin, vmax, bins, num_classes) between providers and consumers.
4. **Shape Integrity**: Lightweight validation of dimensionality and structural properties via `ShapeContract` (e.g., `time_dim=1`).

#### Execution Graph (`core/execution_graph.py`)
The `ExecutionGraph` is an immutable DAG computed once at engine construction time:
1. **Provider Map**: Maps each blackboard path to the component index that produces it.
2. **Dependency Edges**: Derived from `requires`/`provides` contracts (component *i* depends on *j* if *i* requires a path *j* provides).
3. **Topological Sort**: Kahn's algorithm with original-index tiebreaker for deterministic ordering.
4. **Backward-Reachability Pruning**: Walk backward from terminal sinks (`losses.*`, `meta.*`, side-effect components with no provides). Components that don't feed any sink are pruned at build time.
5. **Consumer Map**: For each component, which downstream components read its outputs (used by lazy execution).

```python
# Read-only access for introspection
engine.execution_graph.summary()           # "ExecutionGraph: 8 components → 7 active, 1 pruned"
engine.execution_graph.active_components() # Topologically sorted, pruned list
engine.execution_graph.consumer_map        # {comp_idx: frozenset(downstream_indices)}
engine.execution_graph.terminal_indices    # frozenset of terminal sink indices
```

#### Lazy Execution (`lazy=True`)
Demand-driven per-step resolution. Before each batch, walks the execution order in reverse:
- **Terminal sinks** always execute.
- **Mutating components** (write mode `"overwrite"` or `"append"`) always execute.
- **Pure producers** (write mode `"new"`) are skipped if ALL their demanded outputs are already on the blackboard (e.g. provided by the batch).
- Skipping cascades backward: if B is skipped, A (which only B consumed) may also be skipped.
- Reports `meta["lazy_skipped"]` per step.

#### Blackboard Diffing (`diff=True`)
Per-component change tracking via `core/blackboard_diff.py`. For each component that executes:
- **Added**: New paths that appeared on the blackboard.
- **Removed**: Paths that disappeared (rare).
- **Overwritten**: Paths where the value object changed (detected via `id()` — O(1), no tensor copies).
  - For overwritten tensors: captures `TensorStats` (mean, std, min, max, has_nan, has_inf).

```python
# Access after step
for diff in result["meta"]["blackboard_diffs"]:
    print(diff.detail())
    # [ForwardPassComponent] +2 added
    #   + predictions.values
    #   + predictions.policies
    # [ValueLoss] +1 added, ~1 overwritten
    #   + losses.value_loss
    #   ~ predictions.values  mean=0.0234 std=1.012 [-2.31, 3.14]
```

Cannot loop over network modules. Calls `learner_inference(batch)`, unpacks `LearningOutput`, routes raw tensors to `LossAggregator`.

Algorithm-specific learners/wrappers: `PPOLearner`. DQN-style and supervised learning are now assembled by trainers using `BlackboardEngine` + `TargetBuilder` + `LossAggregator` + callbacks (e.g. `TargetNetworkSyncCallback`).

### `PipelineComponent` (ABC)
```python
# core/component.py
class PipelineComponent(ABC):
    @property
    def requires(self) -> Set[Key]:          # Input contract
    @property
    def provides(self) -> Dict[Key, str]:    # Output contract with write modes
    @property
    def constraints(self) -> list[str]:      # Declarative constraint descriptions
    def validate(self, blackboard) -> None:  # Runtime validation (strict mode)
    def execute(self, blackboard) -> Dict[str, Any]:  # Core logic
```

**Write Modes** (in `provides`):
- `"new"`: Path must not already exist. Eligible for lazy-execution skip.
- `"overwrite"`: Path must already exist. Never skipped by lazy execution.
- `"append"`: Data is added to an existing collection. Never skipped by lazy execution.
- `"optional"`: Path may or may not be produced. Eligible for lazy-execution skip.

Both `requires` and `provides` must be deterministic after `__init__`.

### Contract System (`core/contracts.py`)
```python
Key(
    path: str,                        # Dotted blackboard path (e.g. "predictions.values")
    semantic_type: Type[SemanticType], # What the data means (e.g. ValueEstimate, Policy)
    metadata: Dict[str, Any] = {},    # Representation params (vmin, vmax, bins) — compare=False
    shape: Optional[ShapeContract] = None,  # Structural schema — compare=False
)

ShapeContract(
    ndim: Optional[int] = None,              # Expected tensor rank
    time_dim: Optional[int] = None,          # Axis index of time dimension (typically 1). None = no time.
    event_shape: Optional[Tuple[int, ...]] = None,  # Non-batch, non-time shape (excludes B/T)
)
```

Key identity (hash/equality) is `(path, semantic_type)` only. Metadata and shape are validation-only annotations.

**Parameterized Semantic Types**: Semantic types support structural parameterization via `__class_getitem__`:
```python
ValueEstimate[Scalar]                 # Scalar value estimate
ValueEstimate[Categorical(bins=51)]   # C51 distributional value
Policy[Logits]                  # Raw policy logits
```

Available structures: `Scalar`, `Logits`, `Probs`, `LogProbs`, `Categorical(bins)`, `Quantile(n)`.

Shape validation is opt-in: checks only fire when BOTH provider and consumer declare a field.

### `LossAggregator`
```python
# losses/losses.py
def run(
    predictions: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    context: Dict[str, Any],
    weights: Optional[Tensor] = None,
    gradient_scales: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float], Optional[Tensor]]
# Returns: (total_loss, loss_dict, priorities)
```
`LossModule` subclasses declare `required_predictions` and `required_targets` as sets of string keys. They take raw tensors — **never** `Distribution` objects. Current modules: `StandardDQNLoss`, `C51Loss`, `ValueLoss`, `PolicyLoss`, `RewardLoss`, `ToPlayLoss`, `ConsistencyLoss`, etc.

### `BaseTargetBuilder`
```python
# agents/learners/target_builders.py
def build_targets(batch: Dict[str, Tensor], predictions: LearningOutput, network: nn.Module) -> TargetOutput
```
`TargetOutput` is a dataclass of optional tensors (q_values, value_targets, advantages, policies, rewards, etc.) — use `None` for unused fields, never dummy tensors.

### `ModularReplayBuffer`
```python
# replay_buffers/modular_buffer.py
def store(self, **kwargs)           # Single transition
def sample(self) -> Dict[str, Tensor]
def update_priorities(self, indices, priorities, ids=None)
def set_beta(self, beta: float)
def clear(self)
def share_memory(self)
```
Composed of: `Writer` (CircularWriter / SharedCircularWriter), `Sampler` (UniformSampler / PrioritizedSampler), `InputProcessor`, `OutputProcessor`. Use `buffer_factories.py` to construct algorithm-specific buffers.

### `BaseTrainer`
```python
# agents/trainers/base_trainer.py
def setup(self)
def trigger_test(self, state_dict, step)
def poll_test(self)
@classmethod
def load_from_checkpoint(cls, env, config_class, dir_path, training_step, device, **kwargs)
```

---

## Golden Rules & Anti-Patterns

- **No Dummy Tensors:** Never return `torch.empty(0)` or `torch.zeros()` to satisfy unused dataclass fields. Use `None`.
- **No PyTorch Objects in Loss Functions:** `LossModule.compute_loss` takes raw Tensors — not `Distribution` objects.
- **No `deepcopy`:** Never `copy.deepcopy()` observations. Use `.copy()` for NumPy arrays.
- **Action Masking is Ruthless:** Illegal logits → `-inf`. If Softmax/Gumbel follows, re-apply mask to set illegal probability to exactly `0.0` before sampling.
- **No Python Loops in Samplers/Processors:** `OutputProcessor` and- **No Python loops over tensor dimensions.** Vectorize everything.
- **Structured Semantic Types Required:** Use parameterized types like `ValueEstimate[Scalar]` or `Policy[Categorical(bins=51)]`. String-based mapping (e.g. `dist="scalar"`) is deprecated.
- **No Dummy Configs in Tests:** Never hand-craft `config = {"batch_size": 2}` inline. Always use fixtures from `tests/conftest.py`.

---

## Config System

All configs live in `configs/`. The hierarchy is:

```
ConfigBase (YAML I/O, parse_field, parse_schedule_config)
└── AgentConfig (+ OptimizationConfig, ReplayConfig, RecordConfig, ExecutionConfig mixins)
    ├── RainbowConfig   (+ DistributionalConfig, NoisyConfig, EpsilonGreedyConfig)
    ├── PPOConfig
    └── NFSPConfig
```

Game environments have their own `GameConfig` objects (in `configs/games/`). `AgentConfig.__init__` always takes a `game_config` argument.

Key mixin fields (partial list):
- `OptimizationConfig`: `learning_rate`, `clipnorm`, `weight_decay`, `lr_schedule`, `optimizer`
- `ReplayConfig`: `minibatch_size`, `n_step`, `discount_factor`, `per_alpha`, `per_beta_schedule`
- `SearchConfig`: `search_backend` (`"python"` / `"cpp"` / `"aos"`), `num_simulations`, `search_batch_size`
- `CompilationConfig` (inside `AgentConfig`): controls `torch.compile` and `fullgraph` flag

---

## PyTorch Best Practices

### Tensor Creation
- Always use `device=` kwarg in factory functions: `torch.zeros((10,), device=device)` — never `.cuda()`.
- Never use `torch.Tensor()` (uppercase). Use `torch.tensor()` or factory functions.
- **Forbidden:** `tensor.data` — use `.detach()` instead.
- Use `einops` for all complex reshaping/permuting. Shape comments mandatory if not using einops: `# [B, T, C] -> [B*T, C]`.

### Architecture
- Never use Python `list` for layers — use `nn.ModuleList()` or `nn.Sequential()`.
- Always call `model.eval()` before inference, `model.train()` before training.
- If a `Linear`/`Conv2d` is immediately followed by a normalization layer, set `bias=False`.
- Backbone components (`modules/backbones/`) are created via `BackboneFactory.create(config)`.

### Gradient Management
- Default to `torch.inference_mode()` for inference/MCTS/data generation (faster than `no_grad`).
- Use `optimizer.zero_grad(set_to_none=True)`.

### Compilation & Performance
- `torch.compile` is controlled via `config.compilation`. Disable on MPS. Use `fullgraph=config.compilation.fullgraph` during development to surface graph breaks.
- Apply `torch.compile` to `LossAggregator.run` and return estimators, not just network forward passes.
- Never store attached tensors in the buffer — always `.detach().cpu()` before appending.
- Use `.item()` when logging metrics.
- Keep Replay Buffers on CPU. Move to GPU only immediately before the training step.
- **No Python loops over tensor dimensions.** Vectorize everything.
- Use `channels_last` memory format with AMP + Conv2d (only on supported devices).

### AMP
- `torch.autocast` wraps only the forward pass and loss computation — not the backward pass.
- On MPS: use `float16` (not `bfloat16`). On CPU: avoid `float16` entirely; `bfloat16` only on hardware that supports it.

### Production vs. Debug
- `autograd.detect_anomaly`, profiler, `gradcheck` must be gated behind `config.debug`. `detect_anomaly` slows training 300–500%.
- Disable `torch.compile` first when investigating correctness issues.
- `BlackboardEngine` debug flags (`strict`, `diff`, `lazy`) should be disabled in production. `diff=True` adds snapshot overhead per component; `strict=True` runs `validate()` per component per step.

---

## RL-Specific Rules

- **Target Networks:** Update via `load_state_dict()` or `target.param.data.copy_(online.param.data)`. Never `target = online`.
- **NaN Checks:** Use `torch.autograd.detect_anomaly()` temporarily — remove before production.
- **Seeding (entry point):**
  ```python
  random.seed(seed); np.random.seed(seed)
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
  ```
- **Batch Norm in RL:** Prefer `LayerNorm` — replay buffer sample statistics may not match live environment statistics.
- **Gradient Accumulation:** Use accumulation instead of reducing batch size when GPU memory is tight.
- **Virtual Batching (MCTS):** Keep search tree on CPU. Batch opaque states and call `ModularAgentNetwork` for GPU inference.

---

## Replay Buffer System

`ModularReplayBuffer` is composed of four pluggable components:

| Component | Class | Purpose |
|---|---|---|
| Writer | `CircularWriter` / `SharedCircularWriter` | Manages write indices (FIFO). `SharedCircularWriter` uses PyTorch shared memory for multiprocessing. |
| Sampler | `UniformSampler` / `PrioritizedSampler` | Returns `(indices, weights)`. `PrioritizedSampler` uses a segment tree. |
| InputProcessor | `InputProcessor` subclass | Processes data **before** writing (e.g., compressing observations, accumulating sequences). Returns `None` if still accumulating. |
| OutputProcessor | `OutputProcessor` subclass | Processes raw buffer indices into the final training batch (e.g., N-step returns, GAE). Must be vectorized. |

Use `buffer_factories.py` (`create_dqn_buffer`, etc.) to construct algorithm-specific buffers.

`BufferConfig` declares each stored field: `name`, `shape`, `dtype`, `is_shared`, `fill_value`.

---

## Search Backends

Three MCTS backends selectable via `config.search_backend`:

| Backend | Location | Notes |
|---|---|---|
| `"python"` | `search/search_py/` | Modular pure-Python MCTS. Pluggable scoring, selection, backprop, pruning, root policies. |
| `"aos"` | `search/aos_search/` | Array-of-structures batched MCTS with dynamic masking. |
| `"cpp"` | C++ extension (built via `setup.py`) | Compiled from `search/search_cpp/*.cpp` via pybind11. |

MCTS always lives on CPU.

---

## Ray / Distributed Standards

- **CPU-bound Actors:** Configure `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_PROC_BIND=CLOSE` in `__init__` before importing PyTorch.
- **No blocking `ray.get()` in tight loops.** Use async polling: submit `get_weights.remote()`, check with `timeout=0`.
- **Large objects:** `ref = ray.put(large_obj)` once, pass `ref` to all workers.
- **Parallel tasks:**
  ```python
  futures = [do_work.remote(x) for x in range(4)]  # submit all
  results = ray.get(futures)                         # then wait
  ```
- **No tiny tasks:** Ray overhead ~0.5ms/task. Batch small operations into mega-tasks.
- **Pipelining:** Use `ray.wait()` instead of `ray.get(list)` to avoid straggler blocking.
- **`torch.compile` before spawning workers** — never inside a running actor.
- Executor backends are in `agents/executors/`: `local` (single process) and `torch_mp` (multiprocess with shared memory).

---

## Code Quality

- **No magic values:** Define constants at the top of the file.
- **No `sys.path.append()`:** Project uses `pyproject.toml`. Run as `python -m package.module`.
- **OOP:** Use inheritance (`BlackboardEngine`, `ConfigBase`, `BaseActionSelector`). Inject dependencies — don't instantiate internally.
- **Strong Typing:** All functions must be fully typed (args + return values). Avoid `Any`. Use `if TYPE_CHECKING:` for typing-only imports.
- **Error handling:** Minimize `try/except`. Use `assert` with descriptive messages. No `getattr(obj, 'attr', default)` — use `assert hasattr` then direct access.
- **Docstrings:** New functions must have docstrings. If you touch a function without one, add it.

---

## Testing Standards

### Markers (MANDATORY — top of every test file)
```python
pytestmark = pytest.mark.<unit|integration|slow|regression>
```

### Configuration
- Never define inline `config = {"batch_size": 2}` dicts in tests.
- Use real config objects/fixtures from `tests/conftest.py`.
- Real config instances: `CartPoleConfig`, `PPOConfig`, `RainbowConfig`.

### Structure
```
tests/
├── conftest.py           # All shared fixtures
├── agents/               # Learners, trainers, selectors
├── modules/              # Network components
├── losses/               # Loss functions
├── replay_buffers/       # Buffer, writers, samplers, processors
├── search/               # MCTS
├── learners/             # Learner infrastructure
├── trainers/             # Trainer tests
├── envs/                 # Environment tests
├── configs/              # Config loading
└── regression/           # test_regression_<issue>.py
```
File naming: `test_<component>_<behavior>.py`.

### Style
- Pure pytest functions — no `unittest.TestCase`.
- No global state mutation — use `monkeypatch`.
- Set `torch.manual_seed(42)` and `np.random.seed(42)` for any stochastic test.
- Test the unhappy path with `pytest.raises(ExpectedException)`.

### Pre-flight checklist
- [ ] No `MagicMock` or fake config dicts
- [ ] Real config from `conftest.py`
- [ ] Module-level `pytestmark` at top
- [ ] Seeds set for stochastic tests
- [ ] Pure pytest, no `unittest`

---

## Cross-Platform

- Never assume CUDA. Always use `torch.device` checks.
- No `.sh` scripts for environment setup. Entry points via `python launcher.py <command>` or `python -m`.
- Disable `torch.compile` on MPS. Use `float16` (not `bfloat16`) for AMP on MPS.
- On CPU: avoid `float16`. `bfloat16` only on modern Intel/AMD (AVX-512/AMX) or Apple Silicon.
- On Intel CPUs: set `KMP_BLOCKTIME=1`, `KMP_AFFINITY=granularity=fine,compact,1,0`.
