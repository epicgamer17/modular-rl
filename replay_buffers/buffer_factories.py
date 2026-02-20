import torch
from replay_buffers.modular_buffer import BufferConfig, ModularReplayBuffer
from replay_buffers.concurrency import LocalBackend, TorchMPBackend, ConcurrencyBackend
from typing import Optional
from modules.world_models.inference_output import LearningOutput
from replay_buffers.processors import (
    InputProcessor,
    IdentityInputProcessor,
    NStepInputProcessor,
    MuZeroSequenceInputProcessor,
    PPOInputProcessor,
    StackedInputProcessor,
    LegalMovesInputProcessor,
    ToPlayInputProcessor,
    StandardOutputProcessor,
    MuZeroUnrollOutputProcessor,
    PPOOutputProcessor,
    FilterKeysInputProcessor,
)
from replay_buffers.writers import (
    CircularWriter,
    SharedCircularWriter,
    ReservoirWriter,
    PPOWriter,
)
from replay_buffers.samplers import (
    PrioritizedSampler,
    UniformSampler,
    WholeBufferSampler,
)
from utils.utils import legal_moves_mask


# class RenameKeyInputProcessor(InputProcessor):
#     """
#     Helper processor to map input argument names to buffer names.
#     e.g. 'action' -> 'actions', 'target_policy' -> 'target_policies'
#     """

#     def __init__(self, mapping: dict):
#         self.mapping = mapping

#     def process_single(self, **kwargs):
#         for old_k, new_k in self.mapping.items():
#             if old_k in kwargs and new_k not in kwargs:
#                 kwargs[new_k] = kwargs[old_k]
#         return kwargs
#     def process_single(self, **kwargs):
#         for old_k, new_k in self.mapping.items():
#             if old_k in kwargs and new_k not in kwargs:
#                 kwargs[new_k] = kwargs[old_k]
#         return kwargs


def create_dqn_buffer(
    observation_dimensions,
    max_size,
    num_actions,
    batch_size=32,
    observation_dtype=torch.float32,
    config=None,
    backend: Optional[ConcurrencyBackend] = None,
):
    configs = [
        BufferConfig(
            "observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig(
            "next_observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("dones", shape=(), dtype=torch.bool),
        BufferConfig("next_legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
    ]

    # Standard Pluralization Mapping
    # key_mapping = {
    #     "observation": "observations",
    #     "action": "actions",
    #     "reward": "rewards",
    #     "next_observation": "next_observations",
    #     "done": "dones",
    # }

    if config is not None:
        # N-Step DQN Stack
        # 1. Rename Keys -> 2. Extract Legal Moves -> 3. N-Step Accumulation
        input_stack = StackedInputProcessor(
            [
                # RenameKeyInputProcessor(key_mapping),
                NStepInputProcessor(
                    n_step=config.n_step,
                    gamma=config.discount_factor,
                    num_players=1,
                    reward_key="rewards",
                    done_key="dones",
                ),
                LegalMovesInputProcessor(
                    num_actions,
                    info_key="next_infos",
                    output_key="next_legal_moves_masks",
                ),
                FilterKeysInputProcessor(
                    [
                        "observations",
                        "actions",
                        "rewards",
                        "next_observations",
                        "dones",
                        "next_legal_moves_masks",
                    ]
                ),
            ]
        )

        sampler = PrioritizedSampler(
            max_size,
            alpha=config.per_alpha,
            beta=config.per_beta,
            epsilon=config.per_epsilon,
            max_priority=1.0,
            use_batch_weights=config.per_use_batch_weights,
            use_initial_max_priority=config.per_use_initial_max_priority,
        )
    else:
        # Standard DQN Stack
        print("Creating standard DQN buffer")
        input_stack = StackedInputProcessor(
            [
                LegalMovesInputProcessor(
                    num_actions,
                    info_key="next_infos",
                    output_key="next_legal_moves_masks",
                ),
                FilterKeysInputProcessor(
                    [
                        "observations",
                        "actions",
                        "rewards",
                        "next_observations",
                        "dones",
                        "next_legal_moves_masks",
                    ]
                ),
            ]
        )
        sampler = UniformSampler()

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=configs,
        input_processor=input_stack,
        output_processor=StandardOutputProcessor(),
        writer=CircularWriter(max_size),
        sampler=sampler,
        backend=backend if backend is not None else LocalBackend(),
    )


def create_prioritized_dqn_buffer(
    observation_dimensions,
    max_size,
    num_actions,
    batch_size=32,
    alpha=0.6,
    beta=0.4,
    max_priority=1.0,
    observation_dtype=torch.float32,
    backend: Optional[ConcurrencyBackend] = None,
):
    # Reuse the standard creation logic but swap the sampler
    buffer = create_dqn_buffer(
        observation_dimensions,
        max_size,
        num_actions,
        batch_size,
        observation_dtype,
        backend=backend,
    )
    buffer.sampler = PrioritizedSampler(
        max_size, alpha=alpha, beta=beta, max_priority=max_priority
    )
    return buffer


def create_n_step_buffer(
    observation_dimensions,
    max_size,
    num_actions,
    n_step,
    gamma,
    num_players=1,
    batch_size=32,
    observation_dtype=torch.float32,
    backend: Optional[ConcurrencyBackend] = None,
):
    configs = [
        BufferConfig(
            "observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig(
            "next_observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("next_infos", shape=(), dtype=object),
        BufferConfig("dones", shape=(), dtype=torch.bool),
    ]

    # key_mapping = {
    #     "observation": "observations",
    #     "action": "actions",
    #     "reward": "rewards",
    #     "next_observation": "next_observations",
    #     "next_info": "next_infos",
    #     "done": "dones",
    # }

    # Stack: Rename -> NStep
    input_stack = StackedInputProcessor(
        [
            # RenameKeyInputProcessor(key_mapping),
            NStepInputProcessor(
                n_step,
                gamma,
                num_players,
                reward_key="rewards",
                done_key="dones",
            ),
        ]
    )

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=configs,
        input_processor=input_stack,
        writer=CircularWriter(max_size),
        sampler=UniformSampler(),
        backend=backend if backend is not None else LocalBackend(),
    )


def create_muzero_buffer(
    observation_dimensions,
    max_size,
    num_actions,
    num_players,
    unroll_steps,
    n_step,
    gamma,
    batch_size=32,
    observation_dtype=torch.float32,
    alpha=0.6,
    beta=0.4,
    epsilon=0.01,
    use_batch_weights=True,
    use_initial_max_priority=True,
    lstm_horizon_len=10,
    value_prefix=False,
    tau=0.3,
    multi_process=True,
    backend: Optional[ConcurrencyBackend] = None,
):
    if backend is None:
        backend = TorchMPBackend() if multi_process else LocalBackend()

    configs = [
        BufferConfig(
            "observations",
            shape=observation_dimensions,
            dtype=observation_dtype,
            is_shared=multi_process,
        ),
        BufferConfig("actions", shape=(), dtype=torch.float16, is_shared=multi_process),
        BufferConfig("rewards", shape=(), dtype=torch.float32, is_shared=multi_process),
        BufferConfig("values", shape=(), dtype=torch.float32, is_shared=multi_process),
        BufferConfig(
            "policies",
            shape=(num_actions,),
            dtype=torch.float32,
            is_shared=multi_process,
        ),
        BufferConfig("to_plays", shape=(), dtype=torch.int16, is_shared=multi_process),
        BufferConfig("chances", shape=(1,), dtype=torch.int16, is_shared=multi_process),
        BufferConfig("game_ids", shape=(), dtype=torch.int64, is_shared=multi_process),
        BufferConfig("ids", shape=(), dtype=torch.int64, is_shared=multi_process),
        BufferConfig(
            "training_steps", shape=(), dtype=torch.int64, is_shared=multi_process
        ),
        BufferConfig("terminated", shape=(), dtype=torch.bool, is_shared=multi_process),
        BufferConfig("truncated", shape=(), dtype=torch.bool, is_shared=multi_process),
        BufferConfig("dones", shape=(), dtype=torch.bool, is_shared=multi_process),
        BufferConfig(
            "legal_masks",
            shape=(num_actions,),
            dtype=torch.bool,
            is_shared=multi_process,
        ),
    ]

    # MuZero uses a monolithic processor because it processes the entire Sequence history at once.
    # However, if you wanted to chain post-processing on the 'data' dict returned by process_sequence,
    # you could wrap this in a StackedInputProcessor.
    # For now, MuZeroSequenceInputProcessor handles extraction of policies, values, and legal_masks internally.
    input_processor = MuZeroSequenceInputProcessor(num_actions, num_players)

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=configs,
        input_processor=input_processor,
        output_processor=MuZeroUnrollOutputProcessor(
            unroll_steps,
            n_step,
            gamma,
            num_actions,
            num_players,
            max_size,
            lstm_horizon_len,
            value_prefix,
            tau,
        ),
        writer=(
            SharedCircularWriter(max_size)
            if multi_process
            else CircularWriter(max_size)
        ),
        sampler=PrioritizedSampler(
            max_size,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            use_batch_weights=use_batch_weights,
            use_initial_max_priority=use_initial_max_priority,
            backend=backend,
        ),
        backend=backend,
    )


def create_nfsp_buffer(
    observation_dimensions,
    max_size,
    num_actions,
    batch_size=32,
    observation_dtype=torch.float32,
    backend: Optional[ConcurrencyBackend] = None,
):
    configs = [
        BufferConfig(
            "observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("target_policies", shape=(num_actions,), dtype=torch.float32),
    ]

    # NFSP Stack:
    # 1. LegalMoves: Extract mask from 'info' -> 'legal_moves_masks'
    # 2. Rename: 'observation' -> 'observations', 'target_policy' -> 'target_policies'
    input_stack = StackedInputProcessor(
        [
            LegalMovesInputProcessor(
                num_actions, info_key="info", output_key="legal_moves_masks"
            ),
            FilterKeysInputProcessor(
                ["observations", "legal_moves_masks", "target_policies"]
            ),
        ]
    )

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=configs,
        input_processor=input_stack,
        writer=ReservoirWriter(max_size),
        sampler=UniformSampler(),
        backend=backend if backend is not None else LocalBackend(),
    )


def create_rssm_buffer(
    observation_dimensions,
    max_size,
    batch_length,
    batch_size=32,
    observation_dtype=torch.float32,
    backend: Optional[ConcurrencyBackend] = None,
):
    configs = [
        BufferConfig(
            "observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("actions", shape=(), dtype=torch.float32),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("dones", shape=(), dtype=torch.float32),
    ]

    # RSSM Stack: Simple renaming
    # input_stack = StackedInputProcessor(
    #     [
    # RenameKeyInputProcessor(
    #     {
    #         "observation": "observations",
    #         "action": "actions",
    #         "reward": "rewards",
    #         "done": "dones",
    #     }
    # )
    #     ]
    # )

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=batch_size,
        buffer_configs=configs,
        # input_processor=input_stack,
        writer=CircularWriter(max_size),
        sampler=UniformSampler(),
        backend=backend if backend is not None else LocalBackend(),
    )


def create_ppo_buffer(
    observation_dimensions,
    max_size,
    gamma,
    gae_lambda,
    num_actions,
    observation_dtype=torch.float32,
    backend: Optional[ConcurrencyBackend] = None,
):
    configs = [
        BufferConfig(
            "observations", shape=observation_dimensions, dtype=observation_dtype
        ),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("values", shape=(), dtype=torch.float32),
        BufferConfig("log_probabilities", shape=(), dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("advantages", shape=(), dtype=torch.float32),
        BufferConfig("returns", shape=(), dtype=torch.float32),
    ]

    # PPO Stack:
    # 1. LegalMoves: info -> legal_moves_masks
    # 2. Rename: standard args -> plural keys
    # 3. PPOInputProcessor: (Currently pass-through for single step, but holds finish_trajectory logic)
    input_stack = StackedInputProcessor(
        [
            PPOInputProcessor(gamma, gae_lambda),
            LegalMovesInputProcessor(
                num_actions, info_key="info", output_key="legal_moves_masks"
            ),
        ]
    )

    return ModularReplayBuffer(
        max_size=max_size,
        batch_size=max_size,
        buffer_configs=configs,
        input_processor=input_stack,
        output_processor=PPOOutputProcessor(),
        writer=PPOWriter(max_size),
        sampler=WholeBufferSampler(),
        backend=backend if backend is not None else LocalBackend(),
    )
