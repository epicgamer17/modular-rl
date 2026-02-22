"""
Smoke tests for PPOTrainer to verify initialization and basic training loop.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import gymnasium as gym


def make_cartpole():
    """Factory function for CartPole environment."""
    return gym.make("CartPole-v1")


class MinimalGameConfig:
    """Minimal game config for testing."""

    def __init__(self):
        self.num_players = 1
        self.num_actions = 2
        self.is_discrete = True

    def make_env(self):
        return make_cartpole()


class MinimalActorConfig:
    """Minimal actor config for testing."""

    def __init__(self):
        from torch.optim import Adam

        self.optimizer = Adam
        self.hidden_layers = [64, 64]
        self.clipnorm = 0.5
        self.learning_rate = 0.0003
        self.adam_epsilon = 1e-7


class MinimalCriticConfig:
    """Minimal critic config for testing."""

    def __init__(self):
        from torch.optim import Adam

        self.optimizer = Adam
        self.hidden_layers = [64, 64]
        self.clipnorm = 0.5
        self.learning_rate = 0.0003
        self.adam_epsilon = 1e-7


class MinimalPPOConfig:
    """Minimal PPO config for testing."""

    def __init__(self):
        self.training_steps = 3
        self.steps_per_epoch = 100

        # Architecture Config
        from configs.modules.architecture_config import ArchitectureConfig
        from configs.modules.backbones.dense import DenseConfig
        from configs.modules.heads.base import HeadConfig

        from configs.modules.heads.policy import PolicyHeadConfig

        self.arch = ArchitectureConfig({"activation": "relu", "norm_type": "none"})
        self.backbone = DenseConfig({"widths": [64], "activation": "relu"})
        self.policy_head = PolicyHeadConfig(
            {"neck": None, "output_strategy": {"type": "categorical", "num_classes": 2}}
        )
        self.value_head = HeadConfig(
            {"neck": None, "output_strategy": {"type": "scalar"}}
        )

        from configs.selectors import SelectorConfig

        self.action_selector = SelectorConfig(
            {
                "base": {
                    "type": "categorical",
                    "kwargs": {"exploration": True},
                }
            }
        )

        self.train_policy_iterations = 2
        self.train_value_iterations = 2
        self.minibatch_size = 32
        self.num_minibatches = 2
        self.replay_buffer_size = 200
        self.discount_factor = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.target_kl = 0.02
        self.entropy_coefficient = 0.01
        self.critic_coefficient = 0.5
        self.learning_rate = 0.0003
        self.adam_epsilon = 1e-5
        self.weight_decay = 0.0
        self.kernel_initializer = None
        self.multi_process = False
        self.num_workers = 1
        self.activation = nn.ReLU()
        self.noisy_sigma = 0.0
        self.norm_type = "none"
        self.support_range = None
        self.prob_layer_initializer = None

        # Attributes for NetworkBlock
        self.actor_dense_layer_widths = [64, 64]
        self.critic_dense_layer_widths = [64, 64]

        self.game = MinimalGameConfig()
        self.actor = MinimalActorConfig()
        self.critic = MinimalCriticConfig()


def test_ppo_inference():
    """Test that PPOTrainer select_test_action works with numpy inputs."""
    from agents.trainers.ppo_trainer import PPOTrainer

    config = MinimalPPOConfig()
    # Ensure multi_process is False for simple testing
    config.multi_process = False

    device = torch.device("cpu")
    trainer = PPOTrainer(
        config=config,
        env=make_cartpole(),
        device=device,
        name="ppo_smoke_test",
    )

    print("Running trainer.test(1)...")
    # This calls select_test_action internally, which should handle numpy conversion
    results = trainer.test(1)

    # Explicitly test select_test_action with numpy array to be sure
    import numpy as np

    obs = np.random.randn(4)  # CartPole shape
    print("Testing explicit select_test_action with numpy array...")
    action = trainer.select_test_action(obs, {}, None)

    assert results is not None, "Test results should not be None"
    assert "score" in results, "Test results should contain score"
    print("✓ test_ppo_inference passed")


def test_ppo_trainer_init():
    """Test that PPOTrainer initializes without errors."""
    from agents.trainers.ppo_trainer import PPOTrainer

    config = MinimalPPOConfig()
    device = torch.device("cpu")

    trainer = PPOTrainer(
        config=config,
        env=make_cartpole(),
        device=device,
        name="ppo_smoke_test",
    )

    # Check components are initialized
    assert trainer.agent_network is not None, "Model should be initialized"
    assert trainer.learner is not None, "Learner should be initialized"

    assert trainer.action_selector is not None, "Action selector should be initialized"

    # Clean up
    trainer.executor.stop()
    print("✓ test_ppo_trainer_init passed")


def test_ppo_learner_store():
    """Test that PPOLearner can store transitions."""
    from agents.learners.ppo_learner import PPOLearner
    import numpy as np

    config = MinimalPPOConfig()
    device = torch.device("cpu")

    # Create dummy model
    from modules.agent_nets.ppo import PPONetwork

    model = PPONetwork(
        config=config,
        input_shape=(4,),
    )

    learner = PPOLearner(
        config=config,
        agent_network=model,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=np.float32,
    )

    # Store some transitions
    for i in range(10):
        learner.store(
            observation=np.random.randn(4),
            action=np.random.randint(0, 2),
            value=np.random.randn(),
            log_probability=np.random.randn(),
            reward=np.random.randn(),
        )

    assert learner.pointer == 10, f"Expected pointer=10, got {learner.pointer}"
    print("✓ test_ppo_learner_store passed")


def test_ppo_learner_finish_trajectory():
    """Test that PPOLearner computes GAE correctly."""
    from agents.learners.ppo_learner import PPOLearner
    import numpy as np

    config = MinimalPPOConfig()
    device = torch.device("cpu")

    from modules.agent_nets.ppo import PPONetwork

    model = PPONetwork(
        config=config,
        input_shape=(4,),
    )

    learner = PPOLearner(
        config=config,
        agent_network=model,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=np.float32,
    )

    # Store transitions
    for i in range(5):
        learner.store(
            observation=np.random.randn(4),
            action=np.random.randint(0, 2),
            value=1.0,  # Constant value for testing
            log_probability=-0.5,
            reward=1.0,  # Constant reward
        )

    # Finish trajectory
    learner.finish_trajectory(last_value=0.0)

    # Check advantages and returns were computed
    advantages = learner.advantage_buffer[:5]
    returns = learner.return_buffer[:5]

    assert not torch.all(advantages == 0), "Advantages should be computed"
    assert not torch.all(returns == 0), "Returns should be computed"
    print("✓ test_ppo_learner_finish_trajectory passed")


def test_ppo_learner_step():
    """Test that PPOLearner can run a training step."""
    from agents.learners.ppo_learner import PPOLearner
    import numpy as np

    config = MinimalPPOConfig()
    device = torch.device("cpu")

    from modules.agent_nets.ppo import PPONetwork

    model = PPONetwork(
        config=config,
        input_shape=(4,),
    )

    learner = PPOLearner(
        config=config,
        agent_network=model,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=np.float32,
    )

    # Store enough transitions for a batch
    for i in range(100):
        learner.store(
            observation=np.random.randn(4),
            action=np.random.randint(0, 2),
            value=np.random.randn(),
            log_probability=-0.5,
            reward=np.random.randn(),
        )

    learner.finish_trajectory(last_value=0.0)

    # Run training step
    loss_stats = learner.step()

    assert loss_stats is not None, "Loss stats should be returned"
    assert "policy_loss" in loss_stats, "Policy loss should be tracked"
    assert "value_loss" in loss_stats, "Value loss should be tracked"
    assert "kl_divergence" in loss_stats, "KL divergence should be tracked"
    print("✓ test_ppo_learner_step passed")


if __name__ == "__main__":
    print("Running PPO Trainer verification tests...")
    print()

    test_ppo_learner_store()
    test_ppo_learner_finish_trajectory()
    test_ppo_learner_step()
    test_ppo_trainer_init()
    test_ppo_inference()

    print()
    print("All PPO Trainer tests passed! ✓")
