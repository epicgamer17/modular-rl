import torch
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss


def action_as_onehot(action, num_actions):
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def test_muzero_stochastic_cartpole_smoke():
    """
    Smoke test for STOCHASTIC MuZero on CartPole.
    """
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = {
        "stochastic": True,
        "num_chance": 8,
        "world_model_cls": MuzeroWorldModel,
        "representation_backbone": {"type": "dense", "widths": [32]},
        "dynamics_backbone": {"type": "dense", "widths": [32]},
        "afterstate_dynamics_backbone": {"type": "dense", "widths": [32]},
        "prediction_backbone": {"type": "dense", "widths": [32]},
        "chance_encoder_backbone": {"type": "dense", "widths": [32]},
        "num_simulations": 4,
        "minibatch_size": 2,
        "min_replay_buffer_size": 2,
        "replay_buffer_size": 100,
        "unroll_steps": 2,
        "n_step": 2,
        "multi_process": False,
        "training_steps": 2,
        "games_per_generation": 1,
        "learning_rate": 0.001,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(from_logits=True),
        "reward_loss_function": CategoricalCrossentropyLoss(from_logits=True),
        "policy_loss_function": CategoricalCrossentropyLoss(from_logits=True),
        "support_range": 31,
        "model_name": "smoke_test_stochastic",
    }

    config = MuZeroConfig(config_dict, game_config)

    trainer = MuZeroTrainer(config, env, device=torch.device("cpu"))
    trainer._save_checkpoint = lambda: None

    print("Stochastic MuZero Model Initialized.")
    assert trainer.model is not None
    assert trainer.model.config.stochastic is True
    assert hasattr(trainer.model.world_model, "sigma_head")
    assert hasattr(trainer.model, "afterstate_value_head")

    # Run training loop
    print("Starting Stochastic MuZero Training Loop...")
    trainer.train()
    print("Stochastic MuZero Training Loop Finished.")
    assert trainer.training_step >= 1


if __name__ == "__main__":
    test_muzero_stochastic_cartpole_smoke()
    print("Stochastic MuZero smoke test PASSED!")
