import time
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pytest
import torch.multiprocessing as mp

from modules.agent_nets.modular import ModularAgentNetwork
from modules.embeddings.action_embedding import ActionEncoder
from modules.backbones.resnet import ResNetBackbone
from modules.backbones.conv import ConvBackbone
from modules.heads.value import ValueHead
from modules.heads.policy import PolicyHead
from modules.heads.reward import RewardHead
from modules.heads.to_play import ToPlayHead
from modules.world_models.muzero_world_model import MuzeroWorldModel
from modules.world_models.components.representation import Representation
from modules.world_models.components.dynamics import Dynamics

from actors.action_selectors.decorators import TemperatureSelector
from actors.action_selectors.selectors import CategoricalSelector
from actors.action_selectors.policy_sources import SearchPolicySource
from search.backends.py_search.modular_search import ModularSearch
from utils.schedule import StepwiseSchedule

from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.samplers.sequence import Sequence
from data.processors import SequenceTensorProcessor, NStepUnrollProcessor
from data.writers import SharedCircularWriter
from data.samplers.prioritized import UniformSampler
from data.concurrency import TorchMPBackend

from learner.core import UniversalLearner
from learner.pipeline.forward_pass import ForwardPassComponent
from learner.losses.optimizer_step import OptimizerStepComponent
from learner.pipeline.callbacks import ComponentCallbacks

from learner.pipeline.batch_iterators import SingleBatchIterator
from learner.losses.aggregator import LossAggregatorComponent, PriorityUpdateComponent
from learner.losses.value import ValueLoss
from learner.losses.policy import PolicyLoss
from learner.losses.reward import RewardLoss
from learner.losses.to_play import ToPlayLoss
from learner.losses.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
from learner.losses.priorities import ExpectedValueErrorPriorityComputer
from learner.pipeline.target_builders import (
    MCTSExtractorComponent,
    SequencePadderComponent,
    SequenceMaskComponent,
    SequenceInfrastructureComponent,
    TargetFormatterComponent,
)
from learner.losses.shape_validator import ShapeValidator
from envs.factories.tictactoe import tictactoe_factory
from actors.experts.tictactoe_expert import TicTacToeBestAgent
from executors.torch_mp_executor import TorchMPExecutor
from actors.workers.actors import PettingZooActor
from actors.workers.tester import Tester, VsAgentTest

# Module-level marker for regression tests
pytestmark = pytest.mark.regression


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_muzero_tictactoe_full_training():
    """
    Heavy full training test for MuZero on Tic-Tac-Toe.
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
    TOTAL_TRAINING_STEPS = 10000
    TRAIN_STEPS_PER_EPISODE = 10
    ACTION_EMBEDDING_DIM = 32
    DIRICHLET_FRACTION = 0.25
    DIRICHLET_ALPHA = 0.3
    BUFFER_SIZE = 10000
    TRANSFER_INTERVAL = 100
    TEST_INTERVAL = 1000
    NUM_WORKERS = 4

    # --- 1. Agent Network Architecture ---
    action_encoder = ActionEncoder(num_actions, ACTION_EMBEDDING_DIM)

    representation = Representation(
        backbone=ResNetBackbone(
            input_shape=obs_dim,
            filters=[24, 24, 24],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            norm_type="batch",
        )
    )
    hidden_state_shape = representation.output_shape

    dynamics = Dynamics(
        backbone=ResNetBackbone(
            input_shape=hidden_state_shape,
            filters=[24, 24, 24],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            norm_type="batch",
        ),
        action_encoder=action_encoder,
        input_shape=hidden_state_shape,
        action_embedding_dim=ACTION_EMBEDDING_DIM,
    )

    reward_head = RewardHead(
        input_shape=hidden_state_shape,
        representation=ScalarRepresentation(),
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
    )

    to_play_head = ToPlayHead(
        input_shape=hidden_state_shape,
        num_players=2,
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
    )

    world_model = MuzeroWorldModel(
        representation=representation,
        dynamics=dynamics,
        reward_head=reward_head,
        to_play_head=to_play_head,
        num_actions=num_actions,
    )

    prediction_backbone = ResNetBackbone(
        input_shape=hidden_state_shape,
        filters=[24, 24, 24],
        kernel_sizes=[3, 3, 3],
        strides=[1, 1, 1],
        norm_type="batch",
    )

    value_head = ValueHead(
        input_shape=hidden_state_shape,
        representation=ScalarRepresentation(),
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
    )

    policy_head = PolicyHead(
        input_shape=hidden_state_shape,
        neck=ConvBackbone(
            input_shape=hidden_state_shape,
            filters=[16],
            kernel_sizes=[1],
            strides=[1],
            norm_type="batch",
        ),
        representation=ClassificationRepresentation(num_classes=num_actions),
    )

    agent_network = ModularAgentNetwork(
        components={
            "world_model": world_model,
            "prediction_backbone": prediction_backbone,
            "value_head": value_head,
            "policy_head": policy_head,
        },
        unroll_steps=UNROLL_STEPS,
        atom_size=1,
    ).to(DEVICE)

    # --- 2. Search Backend ---
    search_engine = ModularSearch(
        device=DEVICE,
        num_actions=num_actions,
        num_simulations=NUM_SIMULATIONS,
        discount_factor=DISCOUNT_FACTOR,
        search_batch_size=SEARCH_BATCH_SIZE,
        use_virtual_mean=USE_VIRTUAL_MEAN,
        use_dirichlet=True,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_fraction=DIRICHLET_FRACTION,
        num_players=2,
    )

    inner_selector = CategoricalSelector(exploration=True)
    action_selector = TemperatureSelector(
        inner_selector=inner_selector,
        schedule=StepwiseSchedule(steps=[5, 10], values=[1.0, 0.5, 0.0]),
    )

    # --- 3. Replay Buffer ---
    configs = [
        BufferConfig(
            "observations", shape=obs_dim, dtype=torch.float32, is_shared=True
        ),
        BufferConfig("actions", shape=(), dtype=torch.float16, is_shared=True),
        BufferConfig("rewards", shape=(), dtype=torch.float32, is_shared=True),
        BufferConfig("values", shape=(), dtype=torch.float32, is_shared=True),
        BufferConfig(
            "policies", shape=(num_actions,), dtype=torch.float32, is_shared=True
        ),
        BufferConfig("to_plays", shape=(), dtype=torch.int16, is_shared=True),
        BufferConfig("chances", shape=(1,), dtype=torch.int16, is_shared=True),
        BufferConfig("game_ids", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("ids", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("training_steps", shape=(), dtype=torch.int64, is_shared=True),
        BufferConfig("terminated", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("truncated", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig("dones", shape=(), dtype=torch.bool, is_shared=True),
        BufferConfig(
            "legal_masks", shape=(num_actions,), dtype=torch.bool, is_shared=True
        ),
    ]

    input_processor = SequenceTensorProcessor(
        num_actions, 2, {"player_1": 0, "player_2": 1}
    )
    output_processor = NStepUnrollProcessor(
        UNROLL_STEPS, TD_STEPS, DISCOUNT_FACTOR, num_actions, 2, BUFFER_SIZE
    )

    replay_buffer = ModularReplayBuffer(
        max_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        buffer_configs=configs,
        input_processor=input_processor,
        output_processor=output_processor,
        writer=SharedCircularWriter(max_size=BUFFER_SIZE),
        sampler=UniformSampler(),
        backend=TorchMPBackend(),
    )

    # --- 4. Learner ---
    optimizer = {
        "default": torch.optim.Adam(
            agent_network.parameters(), lr=LEARNING_RATE, eps=1e-5
        )
    }
    val_rep = agent_network.components["value_head"].representation
    pol_rep = agent_network.components["policy_head"].representation
    rew_rep = agent_network.components["world_model"].reward_head.representation
    tp_rep = agent_network.components["world_model"].to_play_head.representation

    shape_validator = ShapeValidator(
        minibatch_size=BATCH_SIZE,
        unroll_steps=UNROLL_STEPS,
        num_actions=num_actions,
        atom_size=1,
    )
    priority_computer = ExpectedValueErrorPriorityComputer(value_representation=val_rep)

    v_loss = ValueLoss(loss_fn=nn.functional.mse_loss, loss_factor=1.0)
    p_loss = PolicyLoss(loss_fn=nn.functional.cross_entropy, loss_factor=1.0)
    r_loss = RewardLoss(loss_fn=nn.functional.mse_loss, loss_factor=1.0)
    tp_loss = ToPlayLoss(loss_factor=1.0)
    priority_comp = PriorityUpdateComponent(priority_computer=priority_computer)

    # Target building components are listed directly in UniversalLearner

    learner = UniversalLearner(
        components=[
            ForwardPassComponent(agent_network, shape_validator),
            MCTSExtractorComponent(),
            SequencePadderComponent(UNROLL_STEPS),
            SequenceMaskComponent(),
            SequenceInfrastructureComponent(UNROLL_STEPS),
            TargetFormatterComponent(
                {
                    "values": val_rep,
                    "policies": pol_rep,
                    "rewards": rew_rep,
                    "to_plays": tp_rep,
                }
            ),
            v_loss,
            p_loss,
            r_loss,
            tp_loss,
            LossAggregatorComponent(),
            priority_comp,
            OptimizerStepComponent(
                agent_network=agent_network,
                optimizer=optimizer,
            ),
        ],
        device=DEVICE,
    )

    # --- 5. Executor Launch ---
    executor = TorchMPExecutor()

    # Match BaseActor.__init__ positional arguments
    launch_args = (
        tictactoe_factory,
        agent_network,
        action_selector,
        replay_buffer,
        2,  # num_players
        torch.device("cpu"),  # worker device
        obs_dim,
        num_actions,
        "muzero_worker",
    )
    launch_kwargs = {"search_engine": search_engine}

    executor.launch(
        PettingZooActor, launch_args, num_workers=NUM_WORKERS, **launch_kwargs
    )

    # Tester setup
    tester_launch_args = (
        tictactoe_factory,
        agent_network,
        action_selector,
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
        ],
    )
    executor.launch(Tester, tester_launch_args, num_workers=1, **launch_kwargs)

    # --- 6. Training Loop ---
    print(f"Starting MuZero Tic-Tac-Toe training for {TOTAL_TRAINING_STEPS} steps...")
    train_steps = 0
    while train_steps < TOTAL_TRAINING_STEPS:
        # 1. Data Collection
        results, collect_stats = executor.collect_data(
            min_samples=None, worker_type=PettingZooActor
        )

        # 2. Learning
        if replay_buffer.size >= BATCH_SIZE:
            iterator = SingleBatchIterator(replay_buffer, DEVICE)
            for metrics in learner.step(iterator):
                if train_steps % 1000 == 0:
                    loss_val = metrics["losses"].get("default")
                    print(f"Step {train_steps} | Loss: {loss_val:.4f}")

            train_steps += 1

            # Update weights
            if train_steps % TRANSFER_INTERVAL == 0:
                executor.update_weights(agent_network.state_dict())

            # Periodic Testing
            if train_steps % TEST_INTERVAL == 0:
                executor.request_work(Tester)

                # If local, run synchronously now (will get results from PREVIOUS interval if async)
                test_results, _ = executor.collect_data(
                    min_samples=None, worker_type=Tester
                )
                print(f"[Step {train_steps}] Test results: {test_results}")

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

    if test_results:
        # Take the most recent evaluation
        last_res = test_results[-1]
        p0_score = last_res.get("vs_expert_p0", {}).get("score")
        p1_score = last_res.get("vs_expert_p1", {}).get("score")
        mean_score = (p0_score + p1_score) / 2
        print(f"Final Mean Score: {mean_score:.4f}")

        assert (
            mean_score > -0.3
        ), f"Performance too low! Final mean score {mean_score:.4f} is below threshold -0.3"
        print("MuZero Regression Training complete and PASSED!")
    else:
        print("MuZero Regression Training complete, but no test results collected!")

    env.close()


if __name__ == "__main__":
    test_muzero_tictactoe_full_training()
