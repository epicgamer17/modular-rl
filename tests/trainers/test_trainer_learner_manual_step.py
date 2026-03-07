import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]

import torch
import torch.nn.functional as F
from agents.learners.muzero_learner import MuZeroLearner
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from modules.agent_nets.modular import ModularAgentNetwork
from modules.world_models.muzero_world_model import MuzeroWorldModel
from torch.optim import Adam
from unittest.mock import MagicMock


def action_as_onehot(action, num_actions):
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def test_learner_directly(make_muzero_config_dict):
    print("Initializing config...")
    game_config = CartPoleConfig()
    config_dict = make_muzero_config_dict(
        world_model_cls=MuzeroWorldModel,
        residual_layers=[],
        representation_dense_layer_widths=[64],
        dynamics_dense_layer_widths=[64],
        actor_dense_layer_widths=[64],
        critic_dense_layer_widths=[64],
        reward_dense_layer_widths=[64],
        actor_conv_layers=[],
        critic_conv_layers=[],
        reward_conv_layers=[],
        to_play_conv_layers=[],
        num_simulations=2,
        minibatch_size=2,
        min_replay_buffer_size=2,
        replay_buffer_size=10,
        unroll_steps=2,
        n_step=2,
        multi_process=False,
        training_steps=1,
        learning_rate=0.001,
        optimizer=Adam,
        action_function=action_as_onehot,
        value_loss_function=F.cross_entropy,
        reward_loss_function=F.cross_entropy,
        policy_loss_function=F.cross_entropy,
        support_range=31,
        action_selector={"base": {"type": "categorical", "kwargs": {}}},
        representation_backbone={"type": "dense", "widths": [64]},
        dynamics_backbone={"type": "dense", "widths": [64]},
        prediction_backbone={"type": "dense", "widths": [64]},
        afterstate_dynamics_backbone={"type": "dense", "widths": [64]},
        projector_hidden_dim=64,
        projector_output_dim=64,
        predictor_hidden_dim=64,
        predictor_output_dim=64,
    )
    config = MuZeroConfig(config_dict, game_config)

    device = torch.device("cpu")
    num_actions = 2
    observation_dimensions = torch.Size((4,))
    support_size = config.support_range * 2 + 1 if config.support_range else 1

    print("Initializing model...")
    model = ModularAgentNetwork(
        config=config,
        num_actions=num_actions,
        input_shape=torch.Size(
            (4,)
        ),  # input_shape usually doesn't include batch in config, handled inside
        channel_first=True,
        world_model_cls=MuzeroWorldModel,
    )

    # Mock learner_inference to return dummy output so we don't need real weights/forward pass
    def mock_learner_inference(batch):
        if isinstance(batch, dict):
            if "obs" in batch:
                B = batch["obs"].shape[0]
            elif "observation" in batch:
                B = batch["observation"].shape[0]
            else:
                # fallback or error
                B = list(batch.values())[0].shape[0]
        else:
            B = batch.obs.shape[0] if hasattr(batch, "obs") else batch.shape[0]
        T = config.unroll_steps

        return {
            "values": F.softmax(torch.randn(B, T + 1, support_size), dim=-1),
            "policies": F.softmax(torch.randn(B, T + 1, num_actions), dim=-1),
            # Pad rewards: (B, T+1, support). First step is dummy.
            "rewards": F.softmax(torch.randn(B, T + 1, support_size), dim=-1),
            "to_plays": torch.randn(B, T + 1, 1),
            "latent_states": torch.randn(B, T + 1, 64),
        }

    model.learner_inference = mock_learner_inference
    model.obs_inference = MagicMock(
        return_value=MagicMock(
            network_state={"dynamics": torch.randn(2, 64), "heads": None}
        )
    )
    model.hidden_state_inference = MagicMock()
    model.afterstate_inference = MagicMock()
    # Mocking check for afterstate inference capability
    model.afterstate_inference = MagicMock()

    preprocess_fn = lambda x: x if torch.is_tensor(x) else torch.tensor(x)

    print("Initializing Learner...")
    learner = MuZeroLearner(
        config=config,
        agent_network=model,
        device=device,
        num_actions=num_actions,
        observation_dimensions=observation_dimensions,
        observation_dtype=torch.float32,
        player_id_mapping={"player_0": 0},
    )

    print("Populating buffer...")
    from replay_buffers.sequence import Sequence

    seq = Sequence(num_players=1)
    seq.append(
        observation=torch.randn(4),
        terminated=False,
        truncated=False,
    )
    for _ in range(10):
        seq.append(
            observation=torch.randn(4),
            terminated=False,
            truncated=False,
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
