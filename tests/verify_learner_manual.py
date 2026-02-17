import torch
import torch.nn.functional as F
from agents.muzero_learner import MuZeroLearner
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole_config import CartPoleConfig
from modules.agent_nets.muzero import AgentNetwork
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss
from torch.optim import Adam
from unittest.mock import MagicMock


def action_as_onehot(action, num_actions):
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def test_learner_directly():
    print("Initializing config...")
    game_config = CartPoleConfig()
    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
        "actor_dense_layer_widths": [64],
        "critic_dense_layer_widths": [64],
        "reward_dense_layer_widths": [64],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "num_simulations": 2,
        "minibatch_size": 2,
        "min_replay_buffer_size": 2,
        "replay_buffer_size": 10,
        "unroll_steps": 2,
        "n_step": 2,
        "multi_process": False,
        "training_steps": 1,
        "learning_rate": 0.001,
        "optimizer": Adam,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(),
        "reward_loss_function": CategoricalCrossentropyLoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": 31,
    }
    config = MuZeroConfig(config_dict, game_config)

    device = torch.device("cpu")
    num_actions = 2
    observation_dimensions = torch.Size((4,))
    support_size = config.support_range * 2 + 1 if config.support_range else 1

    print("Initializing model...")
    model = Network(
        config=config,
        num_actions=num_actions,
        input_shape=torch.Size((config.minibatch_size, 4)),
        channel_first=True,
        world_model_cls=config.world_model_cls,
    )

    # Mock some functions required for loss pipeline
    def predict_initial_inference_fn(x, model):
        B = x.shape[0]
        return (
            F.softmax(torch.randn(B, support_size), dim=-1),  # values
            F.softmax(torch.randn(B, num_actions), dim=-1),  # policies
            torch.randn(B, 64),  # hidden
        )

    def predict_recurrent_inference_fn(
        states, actions_or_codes, reward_h_states=None, reward_c_states=None, model=None
    ):
        B = states.shape[0]
        return (
            F.softmax(torch.randn(B, support_size), dim=-1),  # rewards
            torch.randn(B, 64),  # next_hidden
            F.softmax(torch.randn(B, support_size), dim=-1),  # values
            F.softmax(torch.randn(B, num_actions), dim=-1),  # policies
            torch.randn(B, 1),  # to_play
            reward_h_states,
            reward_c_states,
        )

    def predict_afterstate_recurrent_inference_fn(hidden_states, actions, model):
        B = hidden_states.shape[0]
        return (
            torch.randn(B, 64),
            F.softmax(torch.randn(B, support_size), dim=-1),
            F.softmax(torch.randn(B, 2), dim=-1),
        )

    preprocess_fn = lambda x: x if torch.is_tensor(x) else torch.tensor(x)

    print("Initializing Learner...")
    learner = MuZeroLearner(
        config=config,
        model=model,
        device=device,
        num_actions=num_actions,
        observation_dimensions=observation_dimensions,
        observation_dtype=torch.float32,
        predict_initial_inference_fn=predict_initial_inference_fn,
        predict_recurrent_inference_fn=predict_recurrent_inference_fn,
        predict_afterstate_recurrent_inference_fn=predict_afterstate_recurrent_inference_fn,
        preprocess_fn=preprocess_fn,
    )

    print("Populating buffer...")
    from replay_buffers.sequence import Sequence

    seq = Sequence(num_players=1)
    for _ in range(10):
        seq.append(
            observation=torch.randn(4),
            info={},
            reward=1.0,
            policy=torch.ones(2) / 2,
            value=0.0,
            action=0,
        )
    learner.replay_buffer.store_aggregate(seq)

    print(f"Buffer size: {learner.replay_buffer.size}")

    print("Running learner step...")
    stats = learner.step()
    print(f"Learner step completed! Stats: {stats}")


if __name__ == "__main__":
    test_learner_directly()
