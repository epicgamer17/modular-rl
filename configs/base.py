import torch
import yaml
import os
from modules.utils import (
    prepare_kernel_initializers,
    prepare_activations,
    kernel_initializer_wrapper,
)
import torch.nn.functional as F
from configs.games.game import GameConfig
from configs.modules.compilation import CompilationConfig
from utils.schedule import ScheduleConfig


class ConfigBase:
    def parse_field(
        self, field_name, default=None, wrapper=None, required=True, dtype=None
    ):
        if field_name in self.config_dict:
            val = self.config_dict[field_name]
            print(f"Using         {field_name:30}: {val}")
            if wrapper is not None:
                return wrapper(val)
            return self.config_dict[field_name]

        if default is not None:
            print(f"Using default {field_name:30}: {default}")
            if wrapper is not None:
                return wrapper(default)
            return default

        if required:
            raise ValueError(
                f"Missing required field without default value: {field_name}"
            )
        else:
            print(f"Using         {field_name:30}: {default}")

        if field_name in self._parsed_fields:
            print("warning: duplicate field: ", field_name)
        self._parsed_fields.add(field_name)

    def parse_schedule_config(
        self, field_name: str, defaults: dict = None
    ) -> ScheduleConfig:
        d = self.parse_field(field_name, default=None, required=False)
        if d is None:
            d = {}
        if defaults:
            d = {**defaults, **d}
        print(f"Using         {field_name:30}: {d}")
        return ScheduleConfig.from_dict(d)

    def __init__(self, config_dict: dict, game_config=None):
        self.config_dict = config_dict.copy()
        self.game = game_config
        # Merge legacy nested blocks for backward compatibility
        legacy_blocks = ["architecture", "arch", "search", "replay", "optimization"]
        for block in legacy_blocks:
            if block in self.config_dict and isinstance(self.config_dict[block], dict):
                # Merge block content into top-level, but don't overwrite if top-level already has it
                for k, v in self.config_dict[block].items():
                    if k not in self.config_dict:
                        self.config_dict[k] = v

        self._parsed_fields = set()

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            # Establish a strict contract for how YAML files are formatted
            config_dict = o.get("config_dict", o)
            game = o.get("game", None)

            # Uniform instantiation
            return cls(config_dict=config_dict, game_config=game)

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict)
        if hasattr(self, "game"):
            to_dump["game"] = self.game
        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)

    def __eq__(self, other):
        if not isinstance(other, ConfigBase):
            return False
        if self.config_dict != other.config_dict:
            return False
        if hasattr(self, "game") and hasattr(other, "game"):
            return self.game == other.game
        return True


# --- MODULAR MIXINS ---


class OptimizationConfig:
    def parse_optimization_params(self):
        self.training_steps: int = self.parse_field(
            "training_steps", 10000, wrapper=int
        )
        self.adam_epsilon: float = self.parse_field("adam_epsilon", 1e-8)
        self.momentum = self.parse_field("momentum", 0.9)
        self.learning_rate: float = self.parse_field("learning_rate", 0.001)
        self.clipnorm: int = self.parse_field("clipnorm", 0)
        self.optimizer: torch.optim.Optimizer = self.parse_field(
            "optimizer", torch.optim.Adam
        )
        self.weight_decay: float = self.parse_field("weight_decay", 0.0)
        self.num_minibatches: int = self.parse_field("num_minibatches", 1, wrapper=int)
        self.training_iterations: int = self.parse_field(
            "training_iterations", 1, wrapper=int
        )
        self.lr_schedule = self.parse_schedule_config(
            "lr_schedule",
            defaults={"type": "constant", "initial": self.learning_rate},
        )
        self.test_interval: int = self.parse_field("test_interval", 1000)
        self.checkpoint_interval: int = self.parse_field("checkpoint_interval", 1000)


class ReplayConfig:
    def parse_replay_params(self):
        self.minibatch_size: int = self.parse_field("minibatch_size", 64, wrapper=int)
        self.replay_buffer_size: int = self.parse_field(
            "replay_buffer_size", 5000, wrapper=int
        )
        self.min_replay_buffer_size: int = self.parse_field(
            "min_replay_buffer_size", self.minibatch_size, wrapper=int
        )
        self.n_step: int = self.parse_field("n_step", 1)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.per_alpha: float = self.parse_field("per_alpha", 0.5)
        self.per_beta_schedule = self.parse_schedule_config(
            "per_beta_schedule",
            defaults={
                "type": "linear",
                "initial": 0.5,
                "final": 1.0,
                "decay_steps": 10000,
            },
        )
        self.per_epsilon: float = self.parse_field("per_epsilon", 1e-6)
        self.per_use_batch_weights: bool = self.parse_field(
            "per_use_batch_weights", False
        )
        self.per_use_initial_max_priority: bool = self.parse_field(
            "per_use_initial_max_priority", True
        )
        self.bootstrap_on_truncated: bool = self.parse_field(
            "bootstrap_on_truncated", False
        )
        self.observation_quantization: str = self.parse_field(
            "observation_quantization", None, required=False
        )
        self.observation_compression: str = self.parse_field(
            "observation_compression", None, required=False
        )


class SearchConfig:
    def parse_search_params(self):
        backend_default = self.config_dict.get(
            "search_backend", os.getenv("MCTS_BACKEND", "python")
        )
        self.search_backend: str = str(
            self.parse_field("search_backend", backend_default)
        ).lower()
        self.search_enabled: bool = self.parse_field("search_enabled", True)
        if self.search_backend not in {"python", "cpp", "aos"}:
            raise ValueError(
                f"Unsupported search backend {self.search_backend!r}. Expected 'python', 'cpp', or 'aos'."
            )
        self.known_bounds = self.parse_field(
            "known_bounds", default=None, required=False
        )
        self.min_max_epsilon: float = self.parse_field("min_max_epsilon", 1e-8)

        self.num_simulations: int = self.parse_field("num_simulations", 800)
        self.search_batch_size: int = self.parse_field("search_batch_size", 0)
        self.use_virtual_mean: bool = self.parse_field("use_virtual_mean", False)
        self.virtual_loss: float = self.parse_field("virtual_loss", 3.0)

        self.dirichlet_alpha_adaptive: bool = self.parse_field(
            "dirichlet_alpha_adaptive", False
        )
        self.gumbel: bool = self.parse_field("gumbel", False)
        self.gumbel_m = self.parse_field("gumbel_m", 16)
        self.gumbel_cvisit = self.parse_field("gumbel_cvisit", 50.0)
        # cscale dampens sigma = (cvisit + max_N) * cscale * norm_Q.
        # mctx defaults: 1.0 for perfect-info games (Go/Chess),
        #                0.1 for high-variance games (Atari, Catan).
        self.gumbel_cscale = self.parse_field("gumbel_cscale", 1.0)
        self.pb_c_base: int = self.parse_field("pb_c_base", 19652)
        self.pb_c_init: float = self.parse_field("pb_c_init", 1.25)

        # AOS Search / Additional Search Configs
        self.bootstrap_method: str = self.parse_field(
            "bootstrap_method", "v_mix", required=False
        )
        self.policy_extraction: str = self.parse_field(
            "policy_extraction", "visit_count", required=False
        )
        self.max_search_depth: int = self.parse_field(
            "max_search_depth", self.num_simulations + 1, required=False
        )
        self.max_nodes: int = self.parse_field(
            "max_nodes", self.num_simulations + 1, required=False
        )
        self.num_codes: int = self.parse_field("num_codes", 1, required=False)
        self.use_dirichlet: bool = self.parse_field(
            "use_dirichlet", True, required=False
        )
        self.dirichlet_alpha: float = self.parse_field(
            "dirichlet_alpha", 0.25, required=False
        )
        self.dirichlet_fraction: float = self.parse_field(
            "dirichlet_fraction", 0.25, required=False
        )
        self.use_sequential_halving: bool = self.parse_field(
            "use_sequential_halving", False, required=False
        )
        self.scoring_method: str = self.parse_field(
            "scoring_method", "ucb", required=False
        )
        self.backprop_method: str = self.parse_field(
            "backprop_method", "average", required=False
        )
        self.use_value_prefix: bool = self.parse_field(
            "use_value_prefix", False, required=False
        )
        self.internal_decision_modifier: str = self.parse_field(
            "internal_decision_modifier", "none", required=False
        )
        self.internal_chance_modifier: str = self.parse_field(
            "internal_chance_modifier", "none", required=False
        )


class NoisyConfig:
    def parse_noisy_params(self):
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.0)


class EpsilonGreedyConfig:
    def parse_epsilon_greedy_params(self):
        self.epsilon_schedule = self.parse_schedule_config(
            "epsilon_schedule",
            defaults={"type": "constant", "initial": 0.0},
        )


class ValuePrefixConfig:
    def parse_value_prefix_params(self):
        self.use_value_prefix: bool = self.parse_field("use_value_prefix", False)
        self.lstm_horizon_len: int = self.parse_field("lstm_horizon_len", 5)
        self.lstm_hidden_size: int = self.parse_field("lstm_hidden_size", 64)


class ConsistencyConfig:
    def parse_consistency_params(self):
        self.consistency_loss_factor: float = self.parse_field(
            "consistency_loss_factor", 0.0
        )
        self.projector_output_dim: int = self.parse_field("projector_output_dim", 128)
        self.projector_hidden_dim: int = self.parse_field("projector_hidden_dim", 128)
        self.predictor_output_dim: int = self.parse_field("predictor_output_dim", 128)
        self.predictor_hidden_dim: int = self.parse_field("predictor_hidden_dim", 64)
        assert self.projector_output_dim == self.predictor_output_dim


class RecordConfig:
    def parse_record_params(self):
        self.record_video: bool = self.parse_field("record_video", False)
        self.record_video_interval: int = self.parse_field(
            "record_video_interval", 1000, wrapper=int
        )


class DistributionalConfig:
    def parse_distributional_params(self):
        self.atom_size: int = self.parse_field("atom_size", 1, wrapper=int)
        self.support_range: int = self.parse_field(
            "support_range", None, required=False
        )
        if self.support_range is not None and self.atom_size == 1:
            self.atom_size = self.support_range * 2 + 1
        self.v_min = self.game.min_score
        self.v_max = self.game.max_score


class ExecutionConfig:
    def parse_execution_params(self):
        self.executor_type = self.parse_field("executor_type", "torch_mp")
        self.num_workers = self.parse_field("num_workers", 4, wrapper=int)
        self.num_envs_per_worker = self.parse_field(
            "num_envs_per_worker", 1, wrapper=int
        )
        self.num_puffer_threads = self.parse_field("num_puffer_threads", 2, wrapper=int)
        self.multi_process = self.executor_type != "local"


class Config(
    ConfigBase,
    OptimizationConfig,
    ReplayConfig,
    RecordConfig,
    DistributionalConfig,
    ExecutionConfig,
):
    def __init__(self, config_dict: dict, game_config: GameConfig) -> None:
        super().__init__(config_dict)
        self.game = game_config
        self._verify_game()
        self.save_intermediate_weights: bool = self.parse_field(
            "save_intermediate_weights", False
        )
        self.parse_optimization_params()
        self.parse_replay_params()
        self.parse_record_params()
        self.parse_distributional_params()
        self.parse_execution_params()
        self.loss_function = self.parse_field("loss_function", F.mse_loss)
        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field(
            "kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )
        self.prob_layer_initializer = self.parse_field(
            "prob_layer_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )
        self.norm_type: str = self.parse_field("norm_type", "none")
        self.replay_interval: int = self.parse_field("replay_interval", 1, wrapper=int)

        compilation_dict = self.parse_field("compilation", default={}, required=False)
        self.compilation = CompilationConfig(compilation_dict)

    def _verify_game(self):
        assert (
            self.game is not None
        ), "Config requires a game config to be provided in 'game' field"
        assert (
            self.game.make_env is not None
        ), "Game config must provide a valid environment factory (make_env)"

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            a = cls(config_dict=o["config_dict"], game_config=o.get("game"))
        return a

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict, game=self.game)
        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)


class ActorConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.adam_epsilon = self.parse_field("adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("learning_rate", 0.005)
        self.clipnorm = self.parse_field("clipnorm", None)
        self.optimizer: torch.optim.Optimizer = self.parse_field(
            "optimizer", torch.optim.Adam
        )


class CriticConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.adam_epsilon = self.parse_field("adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("learning_rate", 0.005)
        self.clipnorm = self.parse_field("clipnorm", None)
        self.optimizer: torch.optim.Optimizer = self.parse_field(
            "optimizer", torch.optim.Adam
        )
