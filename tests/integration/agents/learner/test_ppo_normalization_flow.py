import pytest
import numpy as np
import torch

from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig
from replay_buffers.processors import GAEProcessor, AdvantageNormalizer
from replay_buffers.writers import PPOWriter
from replay_buffers.samplers import WholeBufferSampler
from replay_buffers.concurrency import LocalBackend

pytestmark = pytest.mark.integration


def test_ppo_advantage_normalization_flow():
    """
    Tier 2: Integration Test.
    Verifies the flow of advantages from the ReplayBuffer (rollout-level norm)
    to the PPOEpochIterator (mini-batch level norm).
    Checks that advantages are ONLY normalized at the mini-batch level.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    num_steps = 8
    num_actions = 2
    obs_dim = (4,)

    buffer_configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("values", shape=(), dtype=torch.float32),
        BufferConfig("policies", shape=(), dtype=torch.float32),
        BufferConfig("advantages", shape=(), dtype=torch.float32),
        BufferConfig("returns", shape=(), dtype=torch.float32),
        BufferConfig("old_log_probs", shape=(), dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
    ]

    gae = GAEProcessor(gamma=0.99, gae_lambda=0.95)
    normalizer = AdvantageNormalizer()

    replay_buffer = ModularReplayBuffer(
        max_size=num_steps,
        batch_size=num_steps,
        buffer_configs=buffer_configs,
        input_processor=gae,
        output_processor=normalizer,
        writer=PPOWriter(num_steps),
        sampler=WholeBufferSampler(),
        backend=LocalBackend(),
    )

    # Store transitions individually (GAE processes them on sequence store)
    transitions = []
    for i in range(num_steps):
        transitions.append({
            "observations": np.random.randn(*obs_dim).astype(np.float32),
            "actions": np.random.randint(0, num_actions),
            "rewards": float(np.random.randn()),
            "values": float(np.random.randn() * 0.5),
            "policies": float(np.random.randn() * -1.0),  # log_prob stand-in
            "legal_moves_masks": np.ones(num_actions, dtype=bool),
        })

    # Store via aggregate (triggers GAE computation)
    processed = gae.process_sequence(None, transitions=transitions)
    for t in processed["transitions"]:
        replay_buffer.store(**t)

    # Verify raw advantages are NOT zero-mean (GAE doesn't normalize)
    raw_advs = replay_buffer.buffers["advantages"][:num_steps]
    assert not torch.allclose(raw_advs.mean(), torch.tensor(0.0), atol=1e-3)

    # Sample through the normalizer — advantages should now be normalized
    batch = replay_buffer.sample()
    normalized_advs = batch["advantages"]
    assert torch.allclose(normalized_advs.mean(), torch.tensor(0.0), atol=1e-5)


@pytest.mark.unit
def test_gaeprocessor_integration():
    """Tier 1/2: Checks that GAEProcessor produces advantages that PPOBatchProcessor correctly leaves raw."""
    np.random.seed(42)

    gae = GAEProcessor(gamma=0.99, gae_lambda=0.95)

    transitions = [
        {
            "observations": np.zeros(4, dtype=np.float32),
            "actions": 0,
            "rewards": 1.0,
            "values": 0.5,
            "policies": -0.5,
        },
        {
            "observations": np.zeros(4, dtype=np.float32),
            "actions": 1,
            "rewards": 0.0,
            "values": 0.3,
            "policies": -0.7,
        },
    ]

    result = gae.process_sequence(None, transitions=transitions)
    processed = result["transitions"]

    raw_advs = [t["advantages"] for t in processed]

    assert len(raw_advs) == 2
    # Raw GAE advantages should NOT be zero-mean
    assert not np.isclose(np.mean(raw_advs), 0.0, atol=1e-3)
