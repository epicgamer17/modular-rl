import torch
import yaml

from game_configs.game_config import GameConfig
from modules.utils import (
    prepare_kernel_initializers,
    prepare_activations,
)

from losses.basic_losses import MSELoss


class ConfigBase:
    def parse_field(
        self, field_name, default=None, wrapper=None, required=True, dtype=None
    ):
        if field_name in self.config_dict:
            val = self.config_dict[field_name]
            # print("value: ", val)
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

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self._parsed_fields = set()

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            print(o)
            # Handle both nested and flat structures if needed, strictly following original logic:
            if "config_dict" in o:
                config_dict = o["config_dict"]
                game = o.get("game")
            else:
                config_dict = o
                game = None

            # Attempt to init with game_config if the class supports it
            try:
                a = cls(config_dict=config_dict, game_config=game)
            except TypeError:
                a = cls(config_dict=config_dict)

        return a

    def dump(self, filepath: str):
        # Basic dump logic, subclasses might override
        to_dump = dict(config_dict=self.config_dict)
        if hasattr(self, "game"):
            to_dump["game"] = self.game

        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)

    def __eq__(self, other):
        if not isinstance(other, ConfigBase):
            return False
        # Check config dict equality
        if self.config_dict != other.config_dict:
            return False
        # Check game equality if present
        if hasattr(self, "game") and hasattr(other, "game"):
            return self.game == other.game
        return True


# --- MODULAR MIXINS ---


class OptimizationConfig:
    """Standardizes Optimizer, Learning Rate, and Weight Decay parameters."""

    def parse_optimization_params(self):
        # ADD LEARNING RATE SCHEDULES
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
        self.lr_schedule_type: str = self.parse_field("lr_schedule_type", "none")
        self.lr_schedule_steps: list = self.parse_field("lr_schedule_steps", [])
        self.lr_schedule_values: list = self.parse_field("lr_schedule_values", [])


class ReplayConfig:
    """Standardizes Replay Buffer parameters (PER, N-step, Batch Size)."""

    def parse_replay_params(self):
        self.minibatch_size: int = self.parse_field("minibatch_size", 64, wrapper=int)
        self.replay_buffer_size: int = self.parse_field(
            "replay_buffer_size", 5000, wrapper=int
        )
        self.min_replay_buffer_size: int = self.parse_field(
            "min_replay_buffer_size", self.minibatch_size, wrapper=int
        )

        # N-Step & Discount
        self.n_step: int = self.parse_field("n_step", 1)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)

        # Prioritized Replay (PER)
        self.per_alpha: float = self.parse_field("per_alpha", 0.5)
        self.per_beta: float = self.parse_field("per_beta", 0.5)
        self.per_beta_final: float = self.parse_field("per_beta_final", 1.0)
        self.per_epsilon: float = self.parse_field("per_epsilon", 1e-6)
        self.per_use_batch_weights: bool = self.parse_field(
            "per_use_batch_weights", False
        )
        self.per_use_initial_max_priority: bool = self.parse_field(
            "per_use_initial_max_priority", True
        )


class SearchConfig:
    """Standardizes MCTS/Search parameters."""

    def parse_search_params(self):
        self.num_simulations: int = self.parse_field("num_simulations", 800)
        self.search_batch_size: int = self.parse_field("search_batch_size", 0)
        self.use_virtual_mean: bool = self.parse_field("use_virtual_mean", False)
        self.virtual_loss: float = self.parse_field("virtual_loss", 3.0)
        self.root_dirichlet_alpha: float = self.parse_field(
            "root_dirichlet_alpha", 0.25
        )
        self.root_exploration_fraction: float = self.parse_field(
            "root_exploration_fraction", 0.25
        )
        self.root_dirichlet_alpha_adaptive: bool = self.parse_field(
            "root_dirichlet_alpha_adaptive", False
        )

        # Gumbel parameters
        self.gumbel: bool = self.parse_field("gumbel", False)
        self.gumbel_m = self.parse_field("gumbel_m", 16)
        self.gumbel_cvisit = self.parse_field("gumbel_cvisit", 50)
        self.gumbel_cscale = self.parse_field("gumbel_cscale", 1.0)

        # UCB parameters
        self.pb_c_base: int = self.parse_field("pb_c_base", 19652)
        self.pb_c_init: float = self.parse_field("pb_c_init", 1.25)


class NoisyConfig:
    def parse_noisy_params(self):
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.0)


class EpsilonGreedyConfig:
    def parse_epsilon_greedy_params(self):
        self.eg_epsilon: float = self.parse_field("eg_epsilon", 0.00)
        self.eg_epsilon_final: float = self.parse_field("eg_epsilon_final", 0.00)
        self.eg_epsilon_decay_type: str = self.parse_field(
            "eg_epsilon_decay_type", "linear"
        )
        # Defaults to training_steps if not set, needs getattr check
        steps = getattr(self, "training_steps", 10000)
        self.eg_epsilon_final_step: int = self.parse_field(
            "eg_epsilon_final_step", steps
        )


class ValuePrefixConfig:
    """For Value Prefix / LSTM Horizon logic (MuZero/EfficientZero)."""

    def parse_value_prefix_params(self):
        self.value_prefix: bool = self.parse_field("value_prefix", False)
        self.lstm_horizon_len: int = self.parse_field("lstm_horizon_len", 5)
        self.lstm_hidden_size: int = self.parse_field("lstm_hidden_size", 64)


class ConsistencyConfig:
    """For Self-Consistency / EfficientZero params."""

    def parse_consistency_params(self):
        self.consistency_loss_factor: float = self.parse_field(
            "consistency_loss_factor", 0.0
        )
        self.projector_output_dim: int = self.parse_field("projector_output_dim", 128)
        self.projector_hidden_dim: int = self.parse_field("projector_hidden_dim", 128)
        self.predictor_output_dim: int = self.parse_field("predictor_output_dim", 128)
        self.predictor_hidden_dim: int = self.parse_field("predictor_hidden_dim", 64)

        assert self.projector_output_dim == self.predictor_output_dim


class DistributionalConfig:
    """Shared params for Rainbow (atoms) and MuZero (support)."""

    def parse_distributional_params(self):
        self.atom_size: int = self.parse_field("atom_size", 1, wrapper=int)

        # TODO: ADD CHECKS FOR THINGS LIKE PROPER LOSS FUNCTIONS GIVEN SUPPORT SIZES ETC. CHANGING DEFAULTS OF THINGS BASED ON CERTAIN PARAMS (ie change loss function if gumbel = true)
        # assert isinstance(self.config.loss_function, KLDivergenceLoss) or isinstance(
        #     self.config.loss_function, CategoricalCrossentropyLoss
        # ), "Only KLDivergenceLoss and CategoricalCrossentropyLoss are supported for atom_size > 1, recieved {}".format(
        #     self.config.loss_function
        # )

        self.support_range: int = self.parse_field(
            "support_range", None, required=False
        )

        # We can try to grab min/max from game config if not in dict
        game_min = (
            getattr(self.game, "min_score", None)
            if hasattr(self, "game") and self.game
            else None
        )
        game_max = (
            getattr(self.game, "max_score", None)
            if hasattr(self, "game") and self.game
            else None
        )

        # Logic from original RainbowConfig
        self.v_min = game_min
        self.v_max = game_max


def kernel_initializer_wrapper(x):
    if x is None:
        return x
    elif isinstance(x, str):
        return prepare_kernel_initializers(x)
    else:
        assert callable(x)
        return x


class Config(ConfigBase, OptimizationConfig, ReplayConfig):
    """
    Original Config class refactored to use Mixins but preserve identity.
    """

    def __init__(self, config_dict: dict, game_config: GameConfig) -> None:
        self.multi_process = False
        super().__init__(config_dict)
        # could take in a game config and set an action space and observation shape here
        # OR DO THAT IN BASE AGENT?
        self.game = game_config

        self._verify_game()

        # not hyperparameters but utility things
        self.save_intermediate_weights: bool = self.parse_field(
            "save_intermediate_weights", False
        )

        # Initialize Mixins
        self.parse_optimization_params()
        self.parse_replay_params()

        self.loss_function = self.parse_field("loss_function", MSELoss())
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

        self.model_name = self.parse_field("model_name", default="agent")

        self.norm_type: str = self.parse_field("norm_type", "none")
        self.soft_update: bool = self.parse_field("soft_update", False)
        self.min_max_epsilon: float = self.parse_field("min_max_epsilon", 0.01)

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
            print(o)
            a = cls(config_dict=o["config_dict"], game_config=o.get("game"))

        return a

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict, game=self.game)

        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)
