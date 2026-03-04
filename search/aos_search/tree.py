import torch
from dataclasses import dataclass


@dataclass
class FlatTree:
    node_visits: torch.Tensor  # [B, N] int32
    node_values: torch.Tensor  # [B, N] float32 — backprop-updated running mean
    raw_network_values: torch.Tensor  # [B, N] float32 — immutable v̂_π from network
    node_types: torch.Tensor  # [B, N] int8
    node_rewards: torch.Tensor  # [B, N] float32
    to_play: torch.Tensor  # [B, N] int8

    children_index: torch.Tensor
    children_prior_logits: torch.Tensor
    children_action_mask: torch.Tensor
    children_visits: torch.Tensor
    children_rewards: torch.Tensor
    children_values: torch.Tensor

    next_alloc_index: torch.Tensor

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        max_nodes: int,
        num_actions: int,
        num_codes: int,
        device: torch.device,
    ) -> "FlatTree":
        max_edges = max(num_actions, num_codes)

        # [B, N]
        node_visits = torch.zeros(
            (batch_size, max_nodes), dtype=torch.int32, device=device
        )
        node_values = torch.zeros(
            (batch_size, max_nodes), dtype=torch.float32, device=device
        )
        raw_network_values = torch.zeros(
            (batch_size, max_nodes), dtype=torch.float32, device=device
        )
        node_types = torch.zeros(
            (batch_size, max_nodes), dtype=torch.int8, device=device
        )
        to_play = torch.zeros((batch_size, max_nodes), dtype=torch.int8, device=device)
        node_rewards = torch.zeros(
            (batch_size, max_nodes), dtype=torch.float32, device=device
        )

        # [B, N, max_edges]
        children_index = torch.full(
            (batch_size, max_nodes, max_edges), -1, dtype=torch.int32, device=device
        )
        children_prior_logits = torch.full(
            (batch_size, max_nodes, max_edges),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        children_action_mask = torch.ones(
            (batch_size, max_nodes, max_edges), dtype=torch.bool, device=device
        )
        children_visits = torch.zeros(
            (batch_size, max_nodes, max_edges), dtype=torch.int32, device=device
        )
        children_rewards = torch.zeros(
            (batch_size, max_nodes, max_edges), dtype=torch.float32, device=device
        )
        children_values = torch.zeros(
            (batch_size, max_nodes, max_edges), dtype=torch.float32, device=device
        )

        # [B]
        next_alloc_index = torch.ones((batch_size,), dtype=torch.int32, device=device)

        return cls(
            node_visits=node_visits,
            node_values=node_values,
            raw_network_values=raw_network_values,
            node_types=node_types,
            node_rewards=node_rewards,
            to_play=to_play,
            children_index=children_index,
            children_prior_logits=children_prior_logits,
            children_action_mask=children_action_mask,
            children_visits=children_visits,
            children_rewards=children_rewards,
            children_values=children_values,
            next_alloc_index=next_alloc_index,
        )
