import torch
import numpy as np
from configs.agents.rainbow_dqn import RainbowConfig
from agents.learners.rainbow_learner import RainbowLearner


class MockGame:
    def __init__(self):
        self.num_actions = 2
        self.observation_dimensions = (4,)
        self.observation_dtype = torch.uint8
        self.is_discrete = True
        self.min_score = 0.0
        self.max_score = 500.0

    def make_env(self):
        return None


def test_preprocessing():
    device = torch.device("cpu")
    game = MockGame()

    config_dict = {
        "model_name": "debug_preproc",
        "training_steps": 1000,
        "minibatch_size": 2,
        "replay_buffer_size": 100,
        "atom_size": 1,
        "dueling": True,
        "learning_rate": 0.001,
        "action_selector": {
            "base": {"type": "epsilon_greedy", "kwargs": {"epsilon": 0.05}}
        },
        "backbone": {"type": "dense", "hidden_widths": [64]},
        "head": {
            "output_strategy": {"type": "regression"},
            "value_hidden_widths": [64],
            "advantage_hidden_widths": [64],
        },
        "game": game,
        # Mock loss function
        "loss_function": torch.nn.functional.mse_loss,
    }

    # We need a real config object for the learner
    from configs.agents.rainbow_dqn import RainbowConfig

    config = RainbowConfig(config_dict, game)

    # Initialize a dummy model
    from modules.agent_nets.rainbow_dqn import RainbowNetwork

    model = RainbowNetwork(config, output_size=2, input_shape=(4,))
    target_model = RainbowNetwork(config, output_size=2, input_shape=(4,))

    # Initialize Learner
    learner = RainbowLearner(
        config=config,
        model=model,
        target_model=target_model,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.uint8,
    )

    # 1. Test uint8 observation (single)
    obs_uint8 = np.array([0, 127, 255, 64], dtype=np.uint8)
    processed = learner.preprocess(obs_uint8)

    print(f"Processed uint8: {processed}")
    assert processed.dtype == torch.float32, f"Expected float32, got {processed.dtype}"
    assert torch.all(processed >= 0) and torch.all(
        processed <= 1.0
    ), "Normalization failed"
    assert processed.shape == (1, 4), f"Expected shape (1, 4), got {processed.shape}"

    # 2. Test batch of uint8
    batch_uint8 = torch.randint(0, 256, (2, 4), dtype=torch.uint8)
    processed_batch = learner.preprocess(batch_uint8)
    print(f"Processed batch uint8: {processed_batch.shape}, {processed_batch.dtype}")
    assert processed_batch.shape == (2, 4)
    assert processed_batch.max() <= 1.0

    # 3. Test greedy selection using model (mimic Trainer.test)
    state = np.array([0, 127, 255, 64], dtype=np.uint8)
    processed_state = learner.preprocess(state)
    net_out = model.obs_inference(processed_state)
    action = net_out.q_values.argmax(dim=-1)
    print(f"Test action item: {action.item()}")
    assert isinstance(action.item(), int)

    print("SUCCESS: Preprocessing and greedy selection logic verified.")


if __name__ == "__main__":
    test_preprocessing()
