import torch
import torch._dynamo
from typing import Any, Tuple
from configs.games.game import GameConfig
from configs.agents.ppo import PPOConfig
from configs.agents.rainbow_dqn import RainbowConfig
from configs.agents.muzero import MuZeroConfig
from modules.agent_nets.modular import ModularAgentNetwork
from modules.world_models.muzero_world_model import MuzeroWorldModel


def create_mock_game(num_actions: int = 5) -> GameConfig:
    """Creates a mock GameConfig for testing."""
    return GameConfig(
        max_score=100,
        min_score=0,
        is_discrete=True,
        is_image=False,
        is_deterministic=False,
        has_legal_moves=False,
        perfect_information=True,
        multi_agent=False,
        num_players=1,
        num_actions=num_actions,
        make_env=lambda: None,
    )


def debug_graph_breaks(name: str, func: Any, *args: Any) -> bool:
    """Analyzes a function for graph breaks and prints the report."""
    print(f"\n--- Analyzing {name} ---")
    try:
        explanation = torch._dynamo.explain(func)(*args)
        if explanation.graph_break_count == 0:
            print(f"✅ {name}: No graph breaks detected.")
            return True
        else:
            print(
                f"❌ {name}: {explanation.graph_break_count} graph break(s) detected:"
            )
            for i, break_info in enumerate(explanation.graph_breaks):
                print(f"  {i+1}. Reason: {break_info.reason}")
            return False
    except Exception as e:
        print(f"⚠️ {name}: Error focusing on graph breaks: {e}")
        return False


def run_ppo_tests(game: GameConfig, input_shape: Tuple[int, ...], num_actions: int):
    print("\n=== PPO Network Checks ===")
    ppo_config = PPOConfig(
        {
            "policy_head": {"output_strategy": {"type": "categorical"}},
            "value_head": {"output_strategy": {"type": "scalar"}},
            "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
            "actor_config": {"clipnorm": 0},
            "critic_config": {"clipnorm": 0},
            "steps_per_epoch": 128,
        },
        game,
    )
    ppo_net = ModularAgentNetwork(ppo_config, input_shape, num_actions)
    obs = torch.randn(1, *input_shape)

    debug_graph_breaks("PPO obs_inference", ppo_net.obs_inference, obs)
    debug_graph_breaks(
        "PPO learner_inference",
        ppo_net.learner_inference,
        {
            "observations": torch.randn(2, *input_shape),
            "actions": torch.randint(0, num_actions, (2, 1)),
        },
    )


def run_rainbow_tests(game: GameConfig, input_shape: Tuple[int, ...], num_actions: int):
    print("\n=== Rainbow DQN Network Checks ===")
    rainbow_config = RainbowConfig(
        {
            "atom_size": 1,
            "dueling": True,
            "head": {"output_strategy": {"type": "scalar"}},
            "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
        },
        game,
    )
    rainbow_net = ModularAgentNetwork(rainbow_config, input_shape, num_actions)
    obs = torch.randn(1, *input_shape)

    debug_graph_breaks("Rainbow obs_inference", rainbow_net.obs_inference, obs)
    debug_graph_breaks(
        "Rainbow learner_inference",
        rainbow_net.learner_inference,
        {
            "observations": torch.randn(2, *input_shape),
            "actions": torch.randint(0, num_actions, (2, 1)),
            "rewards": torch.randn(2, 1),
            "next_observations": torch.randn(2, *input_shape),
            "terminals": torch.zeros(2, 1),
        },
    )


def run_muzero_tests(game: GameConfig, input_shape: Tuple[int, ...], num_actions: int):
    print("\n=== MuZero Network Checks ===")

    # Deterministic MuZero
    muzero_config = MuZeroConfig(
        {
            "world_model_cls": MuzeroWorldModel,
            "stochastic": False,
            "prediction_backbone": {"type": "identity"},
            "representation_backbone": {"type": "identity"},
            "dynamics_backbone": {"type": "identity"},
            "value_head": {"output_strategy": {"type": "scalar"}},
            "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
            "minibatch_size": 2,
        },
        game,
    )
    muzero_net = ModularAgentNetwork(muzero_config, input_shape, num_actions)
    obs = torch.randn(1, *input_shape)
    outputs = muzero_net.obs_inference(obs)

    debug_graph_breaks("MuZero obs_inference", muzero_net.obs_inference, obs)

    # In MuZero, hidden_state_inference takes an action index tensor (B,)
    action_idx = torch.tensor([0])
    debug_graph_breaks(
        "MuZero hidden_state_inference",
        muzero_net.hidden_state_inference,
        outputs.network_state,
        action_idx,
    )

    debug_graph_breaks(
        "MuZero learner_inference",
        muzero_net.learner_inference,
        {
            "observations": torch.randn(2, *input_shape),
            "actions": torch.randint(0, num_actions, (2, 3)),
            "unroll_observations": torch.randn(2, 4, *input_shape),
        },
    )

    # Stochastic MuZero
    print("\n--- Stochastic MuZero ---")
    stoch_muzero_config = MuZeroConfig(
        {
            "world_model_cls": MuzeroWorldModel,
            "stochastic": True,
            "num_chance": 10,
            "prediction_backbone": {"type": "identity"},
            "representation_backbone": {"type": "identity"},
            "dynamics_backbone": {"type": "identity"},
            "afterstate_dynamics_backbone": {"type": "identity"},
            "chance_encoder_backbone": {"type": "identity"},
            "value_head": {"output_strategy": {"type": "scalar"}},
            "action_selector": {"base": {"type": "categorical", "kwargs": {}}},
            "minibatch_size": 2,
        },
        game,
    )
    stoch_muzero_net = ModularAgentNetwork(
        stoch_muzero_config, input_shape, num_actions
    )
    stoch_outputs = stoch_muzero_net.obs_inference(obs)

    # afterstate_inference takes action index (B,)
    debug_graph_breaks(
        "MuZero afterstate_inference",
        stoch_muzero_net.afterstate_inference,
        stoch_outputs.network_state,
        action_idx,
    )


if __name__ == "__main__":
    input_shape = (4,)
    num_actions = 5
    game_cfg = create_mock_game(num_actions)

    run_ppo_tests(game_cfg, input_shape, num_actions)
    run_rainbow_tests(game_cfg, input_shape, num_actions)
    run_muzero_tests(game_cfg, input_shape, num_actions)
