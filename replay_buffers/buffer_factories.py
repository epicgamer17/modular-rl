import torch
from replay_buffers.modular_buffer import BufferConfig, ModularReplayBuffer
from replay_buffers.concurrency import LocalBackend, TorchMPBackend, ConcurrencyBackend
from typing import Dict, Optional
from replay_buffers.processors import (
    InputProcessor,
    IdentityInputProcessor,
    NStepInputProcessor,
    TerminationFlagsInputProcessor,
    SequenceTensorProcessor,
    GAEProcessor,
    StackedInputProcessor,
    LegalMovesMaskProcessor,
    ToPlayInputProcessor,
    StandardOutputProcessor,
    NStepUnrollProcessor,
    AdvantageNormalizer,
    FilterKeysInputProcessor,
    ObservationCompressionProcessor,
    ObservationDecompressionProcessor,
)
from replay_buffers.writers import (
    CircularWriter,
    SharedCircularWriter,
    ReservoirWriter,
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


class TargetPolicyInputProcessor(InputProcessor):
    """Ensures `target_policies` exists for supervised/NFSP-style buffers.

    Priority:
    1) `target_policies` (already provided)
    2) `target_policy` (legacy singular)
    3) `policies` (from Sequence.policy_history / search metadata)
    4) `actions` -> one-hot
    """

    def __init__(
        self,
        num_actions: int,
        *,
        action_key: str = "actions",
        target_policy_key: str = "target_policies",
    ):
        self.num_actions = num_actions
        self.action_key = action_key
        self.target_policy_key = target_policy_key

    def process_single(self, **kwargs):
        if kwargs.get(self.target_policy_key) is not None:
            return kwargs

        if kwargs.get("target_policy") is not None:
            kwargs[self.target_policy_key] = kwargs["target_policy"]
            return kwargs

        policies = kwargs.get("policies")
        if policies is not None:
            if (
                torch.is_tensor(policies)
                and policies.ndim == 2
                and policies.shape[0] == 1
            ):
                policies = policies.squeeze(0)
            kwargs[self.target_policy_key] = policies
            return kwargs

        action = kwargs.get(self.action_key)
        if action is None:
            return kwargs

        if torch.is_tensor(action):
            action = int(action.item())
        elif isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        one_hot = torch.zeros(self.num_actions, dtype=torch.float32)
        one_hot[action] = 1.0
        kwargs[self.target_policy_key] = one_hot
        return kwargs


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
        BufferConfig("terminated", shape=(), dtype=torch.bool),
        BufferConfig("truncated", shape=(), dtype=torch.bool),
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
                TerminationFlagsInputProcessor(
                    done_key="dones",
                    terminated_key="terminated",
                    truncated_key="truncated",
                ),
                NStepInputProcessor(
                    n_step=config.n_step,
                    gamma=config.discount_factor,
                    num_players=1,
                    reward_key="rewards",
                    done_key="dones",
                ),
                LegalMovesMaskProcessor(
                    num_actions,
                    input_key="next_legal_moves",
                    output_key="next_legal_moves_masks",
                ),
                FilterKeysInputProcessor(
                    [
                        "observations",
                        "actions",
                        "rewards",
                        "next_observations",
                        "terminated",
                        "truncated",
                        "dones",
                        "next_legal_moves_masks",
                    ]
                ),
            ]
        )

        sampler = PrioritizedSampler(
            max_size,
            alpha=config.per_alpha,
            beta=config.per_beta_schedule.initial,
            epsilon=config.per_epsilon,
            max_priority=1.0,
            use_batch_weights=config.per_use_batch_weights,
            use_initial_max_priority=config.per_use_initial_max_priority,
        )
    else:
        # Standard DQN Stack
        input_stack = StackedInputProcessor(
            [
                TerminationFlagsInputProcessor(
                    done_key="dones",
                    terminated_key="terminated",
                    truncated_key="truncated",
                ),
                LegalMovesMaskProcessor(
                    num_actions,
                    input_key="next_legal_moves",
                    output_key="next_legal_moves_masks",
                ),
                FilterKeysInputProcessor(
                    [
                        "observations",
                        "actions",
                        "rewards",
                        "next_observations",
                        "terminated",
                        "truncated",
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

    # NFSP / Supervised Stack:
    # 1. LegalMoves: Extract mask from 'legal_moves' -> 'legal_moves_masks'
    # 2. Ensure `target_policies` exists (from policies or actions)
    input_stack = StackedInputProcessor(
        [
            LegalMovesMaskProcessor(
                num_actions, input_key="legal_moves", output_key="legal_moves_masks"
            ),
            TargetPolicyInputProcessor(num_actions),
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
