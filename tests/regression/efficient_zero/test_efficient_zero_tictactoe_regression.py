import time
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pytest
import torch.multiprocessing as mp

from registries import (
    make_efficient_zero_network,
    make_efficient_zero_search_engine,
    make_efficient_zero_replay_buffer,
    make_efficient_zero_learner,
    make_efficient_zero_actor_engine,
)
from core import SingleBatchIterator
from actors.action_selectors.selectors import ActionSelector
from utils.schedule import StepwiseSchedule
from envs.factories.tictactoe import tictactoe_factory
from components.experts.tictactoe import TicTacToeBestAgent, TicTacToeRandomAgent
from executors.torch_mp_executor import TorchMPExecutor
from executors.workers.actor_worker import ActorWorker
from actors.workers.tester import Tester, VsAgentTest
from utils.plotting import plot_regression_results


# Module-level marker for regression tests
pytestmark = [pytest.mark.regression, pytest.mark.slow]


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_efficient_zero_tictactoe_full_training():
    """
    Heavy full training test for EfficientZero on Tic-Tac-Toe.
    Replicates hyperparameters and logic from muzero_tictactoe.ipynb exactly.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    setup_seeds()

    # --- Exact Hyperparameters from Notebook ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = tictactoe_factory()
    num_actions = env.action_space("player_1").n
    obs_dim = env.observation_space("player_1").shape

    SEARCH_BATCH_SIZE = 5
    USE_VIRTUAL_MEAN = True
    NUM_SIMULATIONS = 25
    DISCOUNT_FACTOR = 0.99
    UNROLL_STEPS = 5
    TD_STEPS = 10
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    TOTAL_TRAINING_STEPS = 20000
    TRAIN_STEPS_PER_EPISODE = 10
    ACTION_EMBEDDING_DIM = 32
    DIRICHLET_FRACTION = 0.25
    DIRICHLET_ALPHA = 0.3
    BUFFER_SIZE = 10000
    TRANSFER_INTERVAL = 100
    TEST_INTERVAL = 1000
    NUM_WORKERS = 4

    # --- 1. Agent Network Architecture ---
    agent_network = make_efficient_zero_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        action_embedding_dim=ACTION_EMBEDDING_DIM,
        resnet_filters=[24, 24, 24],
        unroll_steps=UNROLL_STEPS,
        device=DEVICE,
    )

    # --- 2. Search Backend ---
    search_engine = make_efficient_zero_search_engine(
        num_actions=num_actions,
        num_simulations=NUM_SIMULATIONS,
        discount_factor=DISCOUNT_FACTOR,
        search_batch_size=SEARCH_BATCH_SIZE,
        use_virtual_mean=USE_VIRTUAL_MEAN,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_fraction=DIRICHLET_FRACTION,
        num_players=2,
        device=DEVICE,
    )

    temperature_schedule = StepwiseSchedule(steps=[5, 10], values=[1.0, 0.5, 0.0])

    # --- 3. Replay Buffer ---
    replay_buffer = make_efficient_zero_replay_buffer(
        obs_dim=obs_dim,
        num_actions=num_actions,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        unroll_steps=UNROLL_STEPS,
        td_steps=TD_STEPS,
        discount_factor=DISCOUNT_FACTOR,
        num_players=2,
        player_map={"player_1": 0, "player_2": 1},
    )

    # --- 4. Learner ---
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE, eps=1e-5)

    learner = make_efficient_zero_learner(
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
    actor_engine = make_efficient_zero_actor_engine(
        env=tictactoe_factory(),
        agent_network=agent_network,
        search_engine=search_engine,
        replay_buffer=replay_buffer,
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=2,
        temperature_schedule=temperature_schedule,
        exploration=True,
        device=torch.device("cpu"),
    )

    executor.launch(ActorWorker, (actor_engine,), num_workers=NUM_WORKERS)

    # Tester setup (greedy selector for evaluation - exploration=False forces temp=0)
    test_selector = ActionSelector(input_key="probs")
    tester_launch_args = (
        tictactoe_factory,
        agent_network,
        test_selector,
        None,  # replay_buffer
        2,
        torch.device("cpu"),
        obs_dim,
        num_actions,
        "tester",
        [
            VsAgentTest(
                name="vs_expert_p0",
                num_trials=100,
                opponent=TicTacToeBestAgent(),
                player_idx=0,
            ),
            VsAgentTest(
                name="vs_expert_p1",
                num_trials=100,
                opponent=TicTacToeBestAgent(),
                player_idx=1,
            ),
            VsAgentTest(
                name="vs_random_p0",
                num_trials=100,
                opponent=TicTacToeRandomAgent(),
                player_idx=0,
            ),
            VsAgentTest(
                name="vs_random_p1",
                num_trials=100,
                opponent=TicTacToeRandomAgent(),
                player_idx=1,
            ),
        ],
    )
    executor.launch(
        Tester, tester_launch_args, num_workers=1, search_engine=search_engine
    )

    # --- 6. Training Loop ---
    print(f"Starting EfficientZero Tic-Tac-Toe training for {TOTAL_TRAINING_STEPS} steps...")
    train_steps = 0
    loss_history = {}  # Tracks all individual and total losses
    training_scores = []
    test_score_history = []
    while train_steps < TOTAL_TRAINING_STEPS:

        # 1. Data Collection
        results, _ = executor.collect_data(min_samples=None, worker_type=ActorWorker)
        for res in results:
            if "episode_score" in res:
                # EfficientZero might return multiple players' scores or a mean
                # For TicTacToe, score is usually per player
                if isinstance(res["episode_score"], (list, tuple, np.ndarray)):
                    training_scores.append(np.mean(res["episode_score"]))
                else:
                    training_scores.append(res["episode_score"])

        # 2. Learning
        if replay_buffer.size >= BATCH_SIZE:
            iterator = SingleBatchIterator(replay_buffer, DEVICE)
            for metrics in learner.step(iterator):
                if train_steps % 100 == 0:
                    loss_val = metrics["total_losses"].get("default")
                    individual_losses = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in metrics["losses"].items()]
                    )
                    print(
                        f"Step {train_steps} | Total Loss: {loss_val:.4f} | {individual_losses}"
                    )

                # Collect all individual losses
                for loss_name, loss_val in metrics["losses"].items():
                    if loss_name not in loss_history:
                        loss_history[loss_name] = []
                    loss_history[loss_name].append(loss_val)

                # Collect total loss
                if "total_loss" not in loss_history:
                    loss_history["total_loss"] = []
                loss_history["total_loss"].append(
                    metrics["total_losses"].get("default", 0.0)
                )

            train_steps += 1

            # Update weights and hyperparameters
            if train_steps % TRANSFER_INTERVAL == 0:
                executor.update_weights(
                    agent_network.state_dict(), params={"training_step": train_steps}
                )

            # Periodic Testing
            if train_steps % TEST_INTERVAL == 0:
                executor.request_work(Tester)

                # If local, run synchronously now (will get results from PREVIOUS interval if async)
                test_results, _ = executor.collect_data(
                    min_samples=None, worker_type=Tester
                )
                print(f"[Step {train_steps}] Test results: {test_results}")
                if test_results:
                    last_res = test_results[-1]
                    p0 = last_res.get("vs_expert_p0", {}).get("score", 0.0)
                    p1 = last_res.get("vs_expert_p1", {}).get("score", 0.0)
                    test_score_history.append((p0 + p1) / 2)

                    r0 = last_res.get("vs_random_p0", {}).get("score", 0.0)
                    r1 = last_res.get("vs_random_p1", {}).get("score", 0.0)
                    print(
                        f"[Step {train_steps}] Vs Expert: {(p0+p1)/2:.2f} | Vs Random: {(r0+r1)/2:.2f}"
                    )

    # Final Evaluation
    print("Performing final evaluation...")
    executor.request_work(Tester)

    # Wait for the results to reach the queue
    test_results = []
    for _ in range(300):  # Wait up to 5 minutes for trial completion
        test_results, _ = executor.collect_data(min_samples=None, worker_type=Tester)
        if test_results:
            break
        time.sleep(1.0)

    executor.stop()
    print(f"Final Test results: {test_results}")

    # Plot results
    plot_regression_results(
        name="EfficientZero TicTacToe",
        train_scores=training_scores,
        test_scores=test_score_history,
        losses={k.replace("_", " ").title(): v for k, v in loss_history.items()},
    )

    if test_results:
        # Take the most recent evaluation
        last_res = test_results[-1]
        p0_score = last_res.get("vs_expert_p0", {}).get("score", -1.0)
        p1_score = last_res.get("vs_expert_p1", {}).get("score", -1.0)
        mean_score = (p0_score + p1_score) / 2

        r0_score = last_res.get("vs_random_p0", {}).get("score", -1.0)
        r1_score = last_res.get("vs_random_p1", {}).get("score", -1.0)
        mean_random_score = (r0_score + r1_score) / 2

        print(f"Final Mean Score vs Expert: {mean_score:.4f}")
        print(f"Final Mean Score vs Random: {mean_random_score:.4f}")
    else:
        print("EfficientZero Regression Training complete, but no test results collected!")
        p0_score = p1_score = mean_score = -1.0
        r0_score = r1_score = mean_random_score = -1.0

    # assert to play loss is very very close to 0 (like maybe the mean of last 100 or 1000 is < 0.01)
    if "to_play_loss" in loss_history:
        final_tp_loss = np.mean(loss_history["to_play_loss"][-1000:])
        assert (
            final_tp_loss < 0.01
        ), f"To-play loss is too high! Mean of last 1000: {final_tp_loss:.6f} > 0.01"
        print(f"Final To-Play Loss (L1000): {final_tp_loss:.6f}")

    assert (
        mean_score > 0.15
    ), f"Performance vs Expert too low! Final mean score {mean_score:.4f} is below threshold 0.15"
    assert (
        p0_score > 0.35 and p1_score > -0.05
    ), f"Performance vs Expert too low! Final p0_score {p0_score:.4f} or p1_score {p1_score:.4f} is below threshold"

    # Random Agent Assertions
    assert (
        mean_random_score > 0.8
    ), f"Performance vs Random too low! Final mean score {mean_random_score:.4f} is below threshold 0.8"
    assert (
        r0_score >= 0 and r1_score >= 0
    ), f"Agent lost to Random! p0_score: {r0_score}, p1_score: {r1_score}"

    print("EfficientZero Regression Training complete and PASSED!")

    env.close()


if __name__ == "__main__":
    test_efficient_zero_tictactoe_full_training()
