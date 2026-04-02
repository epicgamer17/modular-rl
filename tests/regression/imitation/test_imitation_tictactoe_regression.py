import random
import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, Any

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.policy import PolicyHead
from agents.learner.base import UniversalLearner
from agents.learner.losses import LossPipeline, ImitationLoss
from agents.learner.target_builders import PassThroughTargetBuilder
from agents.learner.batch_iterators import RepeatSampleIterator
from replay_buffers.modular_buffer import BufferConfig, ModularReplayBuffer
from replay_buffers.processors import (
    StackedInputProcessor,
    LegalMovesMaskProcessor,
    FilterKeysInputProcessor,
)
from replay_buffers.samplers import UniformSampler
from agents.tictactoe_expert import TicTacToeBestAgent
from configs.games.tictactoe import env_factory

def test_imitation_tictactoe_regression():
    """
    Standalone regression test for Imitation Learning on TicTacToe.
    Trains a policy to mimic a TicTacToe expert.
    """
    # --- Hyperparameters ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 1e-3
    TRAINING_STEPS = 500
    BATCH_SIZE = 128
    EXPERT_EPISODES = 200 # Roughly 1000-2000 moves
    
    # --- Setup Environment ---
    env = env_factory(render_mode=None)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # --- Setup Expert ---
    expert = TicTacToeBestAgent()

    # --- Setup Network ---
    # TicTacToe obs from env_factory is [C, H, W] = [8, 3, 3] after stacking
    backbone = MLPBackbone(input_shape=obs_shape, hidden_widths=[128, 128])
    policy_head = PolicyHead(
        input_shape=backbone.output_shape,
        num_actions=num_actions,
        hidden_widths=[64],
    )

    agent_network = ModularAgentNetwork(
        components={
            "feature_block": backbone,
            "policy_head": policy_head,
        },
    ).to(DEVICE)

    # --- Setup Replay Buffer ---
    # We need a buffer that stores observations, legal_moves_masks, and target_policies
    buffer_configs = [
        BufferConfig("observations", shape=obs_shape, dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("target_policies", shape=(num_actions,), dtype=torch.float32),
    ]

    input_processor = StackedInputProcessor([
        LegalMovesMaskProcessor(num_actions, input_key="legal_moves", output_key="legal_moves_masks"),
        FilterKeysInputProcessor(["observations", "legal_moves_masks", "target_policies"]),
    ])

    replay_buffer = ModularReplayBuffer(
        max_size=10000,
        batch_size=BATCH_SIZE,
        buffer_configs=buffer_configs,
        input_processor=input_processor,
        sampler=UniformSampler(),
    )

    # --- Data Collection (Expert) ---
    print(f"Collecting {EXPERT_EPISODES} episodes of expert data...")
    for _ in range(EXPERT_EPISODES):
        env.reset()
        terminated = False
        while not terminated:
            # PettingZoo env: get current player's obs
            obs = env.observe(env.agent_selection)
            info = {"legal_moves": np.where(env.infos[env.agent_selection]["action_mask"])[0]}
            
            # Expert selects action
            action = expert.select_actions(obs, info)
            
            # Store one-hot target policy
            target_policy = np.zeros(num_actions, dtype=np.float32)
            target_policy[action] = 1.0
            
            replay_buffer.store(
                observations=obs,
                legal_moves=info["legal_moves"],
                target_policies=target_policy
            )
            
            env.step(action)
            # Check if game is over
            terminated = any(env.terminations.values()) or any(env.truncations.values())

    print(f"Buffer size: {replay_buffer.size}")

    # --- Setup Learner ---
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE)
    
    # CrossEntropyLoss expects (logits, target_probs) for soft targets in recent PyTorch
    # Or (logits, target_index). Here we use one-hot probs.
    loss_fn = nn.CrossEntropyLoss()
    
    imitation_loss = ImitationLoss(
        device=DEVICE,
        loss_fn=loss_fn,
    )
    
    loss_pipeline = LossPipeline(
        modules=[imitation_loss],
        minibatch_size=BATCH_SIZE,
        num_actions=num_actions,
    )

    # ImitationLoss expects 'policies' in predictions and targets.
    # Our buffer has 'target_policies'. PassThroughTargetBuilder maps it.
    target_builder = PassThroughTargetBuilder(keys=["target_policies"], output_keys=["policies"])

    learner = UniversalLearner(
        agent_network=agent_network,
        device=DEVICE,
        optimizer=optimizer,
        loss_pipeline=loss_pipeline,
        target_builder=target_builder,
    )

    # --- Training Loop ---
    print(f"Starting Imitation Learning training for {TRAINING_STEPS} steps...")
    for step in range(1, TRAINING_STEPS + 1):
        iterator = RepeatSampleIterator(replay_buffer, num_iterations=1, device=DEVICE)
        metrics = []
        for step_metrics in learner.step(iterator):
            metrics.append(step_metrics)
        
        if step % 100 == 0:
            loss_val = np.mean([m["total_loss"] for m in metrics])
            print(f"Step {step} | Loss: {loss_val:.4f}")

    # --- Evaluation ---
    print("\nEvaluating trained policy against random...")
    num_eval_episodes = 50
    wins = 0
    draws = 0
    losses = 0
    
    agent_network.eval()
    for _ in range(num_eval_episodes):
        env.reset()
        terminated = False
        while not terminated:
            curr_agent = env.agent_selection
            obs = env.observe(curr_agent)
            mask = env.infos[curr_agent]["action_mask"]
            
            if curr_agent == "player_0": # Our Agent
                with torch.inference_mode():
                    obs_t = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
                    # For Imitation, we use the policy_head directly
                    result = agent_network.obs_inference(obs_t)
                    # result is InferenceOutput, contains policy_logits
                    logits = result.policy_logits
                    # Apply mask manually
                    logits[0, mask == 0] = -1e9
                    action = logits.argmax().item()
            else: # Random Opponent
                action = env.action_space("player_1").sample(mask)
            
            env.step(action)
            terminated = any(env.terminations.values()) or any(env.truncations.values())
        
        # TicTacToe payoffs in PettingZoo are 1 (win), 0 (draw), -1 (loss)
        final_reward = env.rewards["player_0"]
        if final_reward > 0:
            wins += 1
        elif final_reward == 0:
            draws += 1
        else:
            losses += 1

    print(f"Eval Results (50 games): Wins: {wins}, Draws: {draws}, Losses: {losses}")
    
    assert losses == 0, f"Learned policy lost {losses} games against random!"
    print("Regression test PASSED: Agent plays perfectly against random.")

if __name__ == "__main__":
    test_imitation_tictactoe_regression()
