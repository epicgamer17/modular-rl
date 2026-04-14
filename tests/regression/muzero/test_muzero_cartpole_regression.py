import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pytest
import torch.multiprocessing as mp

from registries import (
    make_muzero_mlp_network,
    make_muzero_search_engine,
    make_muzero_replay_buffer,
    make_muzero_learner,
    make_muzero_actor_engine,
)
from core import SingleBatchIterator, infinite_ticks
from actors.action_selectors.selectors import ActionSelector
from utils.schedule import StepwiseSchedule
from executors.torch_mp_executor import TorchMPExecutor
from executors.workers.actor_worker import ActorWorker
# from utils.plotting import plot_regression_results


# Module-level marker for regression tests
pytestmark = [pytest.mark.regression, pytest.mark.slow]

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_agent(eval_engine, num_episodes=20):
    scores = []
    while len(scores) < num_episodes:
        for result in eval_engine.step(infinite_ticks()):
            meta = result["meta"]
            if "episode_score" in meta:
                scores.append(meta["episode_score"])
                if len(scores) >= num_episodes:
                    return scores
    return scores


def test_muzero_cartpole_full_training():
    """
    Heavy full training test for MuZero on CartPole-v1.
    Replicates hyperparameters and logic from PPO CartPole where applicable.
    """
    from utils.plotting import plot_regression_results
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    setup_seeds()

    # --- Hyperparameters ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENV_ID = "CartPole-v1"

    env = gym.make(ENV_ID)
    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape
    env.close()

    SEARCH_BATCH_SIZE = 5
    USE_VIRTUAL_MEAN = True
    NUM_SIMULATIONS = 25
    DISCOUNT_FACTOR = 0.99
    UNROLL_STEPS = 5
    TD_STEPS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    TOTAL_TRAINING_STEPS = 5000
    ACTION_EMBEDDING_DIM = 16
    DIRICHLET_FRACTION = 0.25
    DIRICHLET_ALPHA = 0.3
    BUFFER_SIZE = 10000
    TRANSFER_INTERVAL = 100
    TEST_INTERVAL = 1000
    NUM_WORKERS = 4
    SUPPORT_RANGE = 500

    # --- 1. Agent Network Architecture ---
    print("Creating Agent Network...")
    agent_network = make_muzero_mlp_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        action_embedding_dim=ACTION_EMBEDDING_DIM,
        hidden_widths=[64, 64],
        support_range=SUPPORT_RANGE,
        unroll_steps=UNROLL_STEPS,
        num_players=1,
        device=DEVICE,
    )

    print("Creating Search Engine...")
    search_engine = make_muzero_search_engine(
        num_actions=num_actions,
        num_simulations=NUM_SIMULATIONS,
        discount_factor=DISCOUNT_FACTOR,
        search_batch_size=SEARCH_BATCH_SIZE,
        use_virtual_mean=USE_VIRTUAL_MEAN,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_fraction=DIRICHLET_FRACTION,
        num_players=1,
        device=DEVICE,
    )

    temperature_schedule = StepwiseSchedule(steps=[15, 30], values=[1.0, 0.5, 0.0])

    print("Creating Replay Buffer...")
    # Cartpole is 1-player
    replay_buffer = make_muzero_replay_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        unroll_steps=UNROLL_STEPS,
        td_steps=TD_STEPS,
        discount_factor=DISCOUNT_FACTOR,
        num_players=1,
        player_map=None,
    )

    print("Creating Learner...")
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE, eps=1e-5)

    learner = make_muzero_learner(
        agent_network=agent_network,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        unroll_steps=UNROLL_STEPS,
        num_actions=num_actions,
        device=DEVICE,
    )

    # --- 5. Executor Launch ---
    # Share model parameters across processes so optimizer updates are visible to workers
    agent_network.share_memory()

    executor = TorchMPExecutor()

    # Create Actor Engine
    actor_engine = make_muzero_actor_engine(
        env=gym.make(ENV_ID),
        agent_network=agent_network,
        search_engine=search_engine,
        replay_buffer=replay_buffer,
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=1,
        temperature_schedule=temperature_schedule,
        exploration=True,
        device=torch.device("cpu"),
    )

    print("Launching Executor...")
    executor.launch(ActorWorker, (actor_engine,), num_workers=NUM_WORKERS)
    print("Executor Launched.")

    print("Creating Evaluation Engine...")
    # Evaluator Engine
    eval_engine = make_muzero_actor_engine(
        env=gym.make(ENV_ID),
        agent_network=agent_network,
        search_engine=search_engine,
        replay_buffer=None,
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=1,
        temperature_schedule=None,  # Actually we want exploration=False
        exploration=False,
        device=torch.device("cpu"),
    )

    # --- 6. Training Loop ---
    print(f"Starting MuZero CartPole training for max {TOTAL_TRAINING_STEPS} steps...")
    train_steps = 0
    loss_history = {}
    training_scores = []
    test_score_history = []

    # Let the replay buffer fill a bit before starting
    print("Waiting for replay buffer to fill...")
    time.sleep(2)

    while train_steps < TOTAL_TRAINING_STEPS:
        if train_steps == 0:
             print(f"Waiting for buffer to fill... Current size: {replay_buffer.size}/{BATCH_SIZE}", end="\r")
        
        # 1. Data Collection
        results, _ = executor.collect_data(min_samples=None, worker_type=ActorWorker)
        for res in results:
            if "episode_score" in res:
                # Some dicts might return arrays
                if isinstance(res["episode_score"], (list, tuple, np.ndarray)):
                    # For a single-player game it might be [score]
                    training_scores.append(res["episode_score"][0])
                else:
                    training_scores.append(res["episode_score"])

        # 2. Learning
        if replay_buffer.size >= BATCH_SIZE:
            iterator = SingleBatchIterator(replay_buffer, DEVICE)
            for metrics in learner.step(iterator):
                # Collect losses
                for loss_name, loss_val in metrics["losses"].items():
                    if loss_name not in loss_history:
                        loss_history[loss_name] = []
                    loss_history[loss_name].append(loss_val)

                if "total_loss" not in loss_history:
                    loss_history["total_loss"] = []
                loss_history["total_loss"].append(
                    metrics["total_losses"].get("default", 0.0)
                )

                if train_steps % 100 == 0:
                    loss_val = metrics["total_losses"].get("default")
                    individual_losses = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in metrics["losses"].items()]
                    )
                    recent_score = (
                        np.mean(training_scores[-100:])
                        if len(training_scores) > 0
                        else 0.0
                    )
                    print(
                        f"Step {train_steps} | L100 Score: {recent_score:.2f} | Total Loss: {loss_val:.4f} | {individual_losses}"
                    )

            train_steps += 1

            # Update weights and hyperparameters
            if train_steps % TRANSFER_INTERVAL == 0:
                executor.update_weights(
                    agent_network.state_dict(), params={"training_step": train_steps}
                )

            # Periodic Testing
            if train_steps % TEST_INTERVAL == 0:
                print("Running evaluation...")
                eval_scores = evaluate_agent(eval_engine, num_episodes=5)
                mean_eval_score = np.mean(eval_scores)
                test_score_history.append(mean_eval_score)
                print(
                    f"[Step {train_steps}] Evaluated {len(eval_scores)} episodes: {eval_scores} | Mean: {mean_eval_score:.2f}"
                )

        # Check early stopping
        if len(training_scores) >= 100 and np.mean(training_scores[-100:]) >= 450.0:
            print(
                f"Solved early at step {train_steps} with average score {np.mean(training_scores[-100:]):.2f}"
            )
            break

    # Final Evaluation
    print("Performing final evaluation...")
    eval_scores = evaluate_agent(eval_engine, num_episodes=20)
    final_eval_score = np.mean(eval_scores)
    test_score_history.append(final_eval_score)
    print(f"Final Test results: {eval_scores} | Mean: {final_eval_score:.2f}")

    executor.stop()

    # Plot results
    plot_regression_results(
        name="MuZero CartPole",
        train_scores=training_scores,
        test_scores=test_score_history,
        losses={k.replace("_", " ").title(): v for k, v in loss_history.items()},
    )

    # Check that performance is adequate
    assert (
        final_eval_score >= 450.0
    ), f"Performance too low! Final mean eval score {final_eval_score:.2f} is below 450.0"

    print("MuZero CartPole Regression Training complete and PASSED!")


if __name__ == "__main__":
    test_muzero_cartpole_full_training()
