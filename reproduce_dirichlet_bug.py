import torch
import numpy as np
from search.aos_search.tree import FlatTree
from search.aos_search.batched_mcts import batched_mcts_step
from dataclasses import dataclass


@dataclass
class MockOutput:
    value: torch.Tensor
    reward: torch.Tensor
    to_play: torch.Tensor
    policy: any


@dataclass
class MockPolicy:
    logits: torch.Tensor


class MockNetwork:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def hidden_state_inference(self, parent_indices, actions):
        B = parent_indices.shape[0]
        # Return all zero logits
        return MockOutput(
            value=torch.zeros(B, device=parent_indices.device),
            reward=torch.zeros(B, device=parent_indices.device),
            to_play=torch.zeros(B, dtype=torch.int8, device=parent_indices.device),
            policy=MockPolicy(
                logits=torch.zeros((B, self.num_actions), device=parent_indices.device)
            ),
        )


def reproduce():
    device = torch.device("cpu")
    num_actions = 4
    B = 1

    # 1. Allocate tree
    tree = FlatTree.allocate(
        batch_size=B, max_nodes=100, num_actions=num_actions, num_codes=1, device=device
    )

    # 2. Initialize root (node 0)
    # Set root priors to all 1.0 (logits)
    root_logits = torch.ones((B, num_actions), device=device)
    tree.children_prior_logits[:, 0, :num_actions] = root_logits
    tree.node_visits[:, 0] = 1
    tree.to_play[:, 0] = 0

    # 3. Define modifier (e.g. all 10s)
    def tag_modifier(logits):
        return torch.full_like(logits, 10.0)

    # 4. Scenario A: Default behavior (No modifiers)
    print("\n--- Scenario A: Default (No Modifiers) ---")
    network = MockNetwork(num_actions)
    batched_mcts_step(
        tree=tree,
        agent_network=network,
        max_depth=5,
        pb_c_init=1.25,
        pb_c_base=19652,
        discount=0.99,
        search_batch_size=1,
    )

    # Selection might have picked any action. Let's find any child of root.
    child_indices = tree.children_index[0, 0, :num_actions]
    expanded_child_idx = -1
    for a in range(num_actions):
        if child_indices[a] != -1:
            expanded_child_idx = child_indices[a].item()
            break

    if expanded_child_idx != -1:
        child_priors = tree.children_prior_logits[0, expanded_child_idx, :num_actions]
        print(f"Child priors (Node {expanded_child_idx}): {child_priors}")
        if torch.allclose(
            child_priors, torch.zeros(num_actions)
        ):  # MockNetwork returns zeros
            print("SUCCESS: Child priors match raw network output (no poisoning)")
        else:
            print(f"FAILURE: Child priors were unexpectedly modified: {child_priors}")
    else:
        print("FAILURE: No child was expanded in Scenario A")

    # 5. Scenario B: Explicit internal modifier
    print("\n--- Scenario B: Explicit Internal Modifier ---")
    # Reset expansion for Node 2
    # We'll just run another step and see a new allocation
    batched_mcts_step(
        tree=tree,
        agent_network=network,
        max_depth=5,
        pb_c_init=1.25,
        pb_c_base=19652,
        discount=0.99,
        search_batch_size=1,
        decision_modifier_fn=tag_modifier,
    )

    # Find the next expanded child (it should be at index 2 if node 1 was expanded before)
    # Actually children_index for root node 0 will have multiple children if we run more simulations.
    # But selection might pick the same one. Let's look at next_alloc_index.
    new_node_idx = tree.next_alloc_index[0].item() - 1
    new_node_priors = tree.children_prior_logits[0, new_node_idx, :num_actions]
    print(f"New node priors (Node {new_node_idx}): {new_node_priors}")
    if torch.allclose(new_node_priors, torch.full((num_actions,), 10.0)):
        print("SUCCESS: Internal modifier correctly applied when explicitly provided")
    else:
        print("FAILURE: Internal modifier NOT applied")


if __name__ == "__main__":
    reproduce()
