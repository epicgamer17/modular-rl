import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
import numpy as np
from search import set_backend, ModularSearch, get_backend_name
from modules.models.inference_output import InferenceOutput

# Fallback for benchmark fixture if pytest-benchmark is not installed
@pytest.fixture
def perf_benchmark(request):
    try:
        return request.getfixturevalue("benchmark")
    except (pytest.FixtureLookupError, Exception):
        # Fallback simple timer
        def _benchmark(func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"\n[DUMMY BENCHMARK] {func.__name__} took {end-start:.6f}s")
            return result
        return _benchmark

pytestmark = pytest.mark.performance

class MockAgentNetwork(nn.Module):
    """Mock AgentNetwork with controllable latency and input shape."""
    def __init__(self, num_actions=9, hidden_state_size=64, latency=0.0):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_state_size = hidden_state_size
        self.latency = latency
        self.input_shape = (3, 3) # TicTacToe observation
        self.device = torch.device("cpu")

    def obs_inference(self, obs):
        """Mock root inference with latency."""
        if self.latency > 0:
            time.sleep(self.latency)
        batch_size = obs.shape[0] if torch.is_tensor(obs) else len(obs)
        return InferenceOutput(
            value=torch.zeros(batch_size),
            policy=Categorical(logits=torch.zeros(batch_size, self.num_actions)),
            recurrent_state={"state": torch.zeros(batch_size, self.hidden_state_size)},
            to_play=torch.zeros(batch_size, dtype=torch.long)
        )

    def hidden_state_inference(self, hidden_state, action):
        """Mock recurrent inference with latency."""
        if self.latency > 0:
            time.sleep(self.latency)
        batch_size = action.shape[0]
        return InferenceOutput(
            value=torch.zeros(batch_size),
            policy=Categorical(logits=torch.zeros(batch_size, self.num_actions)),
            recurrent_state={"state": torch.zeros(batch_size, self.hidden_state_size)},
            reward=torch.zeros(batch_size),
            to_play=torch.zeros(batch_size, dtype=torch.long)
        )
    
    def afterstate_inference(self, hidden_state, code):
        """Mock afterstate inference for stochastic MCTS."""
        if self.latency > 0:
            time.sleep(self.latency)
        batch_size = code.shape[0]
        return InferenceOutput(
            value=torch.zeros(batch_size),
            policy=Categorical(logits=torch.zeros(batch_size, self.num_actions)),
            recurrent_state={"state": torch.zeros(batch_size, self.hidden_state_size)},
            to_play=torch.zeros(batch_size, dtype=torch.long)
        )

@pytest.fixture
def benchmark_config():
    """Generic configuration for MCTS benchmarking."""
    class SubConfig: 
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class SimpleConfig:
        def __init__(self):
            # General
            self.num_simulations = 100
            self.max_search_depth = 10
            self.max_nodes = 512
            self.search_batch_size = 1
            self.pb_c_init = 1.25
            self.pb_c_base = 19652
            self.discount_factor = 1.0
            self.bootstrap_method = "value"
            self.known_bounds = None
            self.min_max_epsilon = 0.01
            self.policy_extraction = "visit_count"
            self.stochastic = False
            self.scoring_method = "ucb"
            self.backprop_method = "average"
            
            # Gumbel/Dirichlet
            self.use_dirichlet = False
            self.dirichlet_alpha = 0.3
            self.dirichlet_fraction = 0.25
            self.gumbel_m = 4
            self.gumbel_cvisit = 50
            self.gumbel_cscale = 1.0
            self.use_sequential_halving = False
            
            # Stochastic / codes
            self.num_codes = 0
            self.virtual_loss = 1.0
            self.use_virtual_mean = False
            
            # Sub-configs
            self.game = SubConfig(num_actions=9, num_players=2)
            self.compilation = SubConfig(enabled=False, fullgraph=False)
            self.internal_decision_modifier = "none"
            self.internal_chance_modifier = "none"
            
            # MuZero support
            self.support_range = None
            
            # For backward compatibility with other backends
            self.gumbel = False
            self.lstm_horizon_len = 5
            
    return SimpleConfig()

@pytest.mark.parametrize("backend", ["python", "cpp", "aos"])
def test_non_vectorized_backend_throughput(perf_benchmark, backend, benchmark_config):
    """Compare pure MCTS speed (unbatched) across Python, C++, and AOS backends."""
    try:
        set_backend(backend)
    except (ImportError, ValueError):
        pytest.skip(f"Backend {backend} not available or failed to load")
    num_actions = benchmark_config.game.num_actions
    agent = MockAgentNetwork(num_actions=num_actions)
    search = ModularSearch(benchmark_config, device=torch.device("cpu"), num_actions=num_actions)
    obs = torch.zeros(1, 3, 3) 
    info = {"player": 0, "legal_moves": [list(range(num_actions))]}
    for _ in range(3): search.run(obs, info, agent)
    perf_benchmark(search.run, obs, info, agent)

@pytest.mark.parametrize("backend", ["python", "cpp", "aos"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_vectorized_backend_throughput(perf_benchmark, backend, batch_size, benchmark_config):
    """Compare batched MCTS speed across backends for different batch sizes."""
    try:
        set_backend(backend)
    except (ImportError, ValueError):
        pytest.skip(f"Backend {backend} not available or failed to load")
    num_actions = benchmark_config.game.num_actions
    agent = MockAgentNetwork(num_actions=num_actions)
    search = ModularSearch(benchmark_config, device=torch.device("cpu"), num_actions=num_actions)
    batched_obs = torch.zeros(batch_size, 3, 3)
    batched_info = {"player": torch.zeros(batch_size, dtype=torch.long), "legal_moves": [list(range(num_actions))] * batch_size}
    for _ in range(3): search.run_vectorized(batched_obs, batched_info, agent)
    perf_benchmark(search.run_vectorized, batched_obs, batched_info, agent)

@pytest.mark.parametrize("backend", ["cpp", "aos"])
@pytest.mark.parametrize("latency", [0.0, 0.001, 0.01])
def test_search_throughput_scaling_with_nn_latency(perf_benchmark, backend, latency, benchmark_config):
    """Benchmark how search throughput behaves as NN inference time increases."""
    try:
        set_backend(backend)
    except (ImportError, ValueError):
        pytest.skip(f"Backend {backend} not available or failed to load")
    num_actions = benchmark_config.game.num_actions
    agent = MockAgentNetwork(num_actions=num_actions, latency=latency)
    search = ModularSearch(benchmark_config, device=torch.device("cpu"), num_actions=num_actions)
    batch_size = 4
    batched_obs = torch.zeros(batch_size, 3, 3)
    batched_info = {"player": torch.zeros(batch_size, dtype=torch.long), "legal_moves": [list(range(num_actions))] * batch_size}
    perf_benchmark(search.run_vectorized, batched_obs, batched_info, agent)

@pytest.mark.parametrize("backend", ["cpp", "aos"])
def test_multiprocess_vs_vectorized_comparison(perf_benchmark, backend, benchmark_config):
    """Compare 1 vectorized search vs 4 parallel individual searches."""
    try:
        set_backend(backend)
    except (ImportError, ValueError):
        pytest.skip(f"Backend {backend} not available or failed to load")
    num_actions = benchmark_config.game.num_actions
    num_workers = 4
    agent = MockAgentNetwork(num_actions=num_actions)
    batched_obs = torch.zeros(num_workers, 3, 3)
    batched_info = {"player": torch.zeros(num_workers, dtype=torch.long), "legal_moves": [list(range(num_actions))] * num_workers}
    search = ModularSearch(benchmark_config, device=torch.device("cpu"), num_actions=num_actions)

    def run_vectorized_batch(): search.run_vectorized(batched_obs, batched_info, agent)
    print(f"\nBenchmarking {backend} backend:")
    perf_benchmark(run_vectorized_batch)
