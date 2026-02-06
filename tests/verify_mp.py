from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.cartpole_config import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss
import torch
import time


def action_as_onehot(action, num_actions):
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def verify():
    print("Starting verification of MuZero multiprocessing...")
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
        "actor_dense_layer_widths": [16],
        "critic_dense_layer_widths": [16],
        "reward_dense_layer_widths": [16],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "num_simulations": 1,
        "minibatch_size": 1,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 10,
        "unroll_steps": 1,
        "n_step": 1,
        "multi_process": True,
        "num_workers": 1,
        "training_steps": 1,
        "games_per_generation": 1,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(),
        "reward_loss_function": CategoricalCrossentropyLoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": 31,
    }

    config = MuZeroConfig(config_dict, game_config)
    agent = MuZeroAgent(env, config, name="verify_mp", device="cpu")

    print("Initial MRO:", MuZeroAgent.__mro__)

    try:
        print("Starting workers...")
        stats_client = agent.stats.get_client()
        agent.start_workers(
            worker_fn=agent.worker_fn, num_workers=1, stats_client=stats_client
        )

        print("Workers started. Waiting 5 seconds for some self-play...")
        time.sleep(5)

        print("Checking for errors...")
        agent.check_worker_errors()

        print("Stats steps:", agent.stats.get_num_steps())

        print("Stopping workers...")
        agent.stop_workers()
        print("Workers stopped successfully.")
        print("Verification PASSED!")

    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback

        traceback.print_exc()
        if hasattr(agent, "stop_workers"):
            agent.stop_workers()
        exit(1)


if __name__ == "__main__":
    verify()
