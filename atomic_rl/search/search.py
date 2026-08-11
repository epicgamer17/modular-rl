import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise
from .backpropagation import backpropagate_
from .expansion import expand_node_
from .selection import select_leaf
from .tree import init_mcts_tree
from .qtransforms import qtransform_by_parent_and_siblings
from .policies import get_mcts_visit_policy


# TODO: remember we eventually want gumbel sequential halving and possibly other search methods too.
# TODO: dont hard code dirichlet params, pass em in as optional arguments
# TODO: avoid flags like is_zero_sum
# TODO: make it easier to do stochastic muzero and stochastic alphazero.
# TODO: should this just return the tree?
# TODO: what about sampled muzero?
# TODO: make this less hardcoded. I don't mind if search orchestration goes in the orchestration code, if it gives users more freedom.
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, Optional
from ..utils import add_dirichlet_noise
from .backpropagation import backpropagate_
from .expansion import expand_node_
from .selection import select_leaf
from .tree import init_mcts_tree
from .qtransforms import qtransform_by_parent_and_siblings
from .policies import get_mcts_visit_policy


def mcts_search(
    root_embeddings: torch.Tensor,
    root_logits: torch.Tensor,
    root_value: torch.Tensor,
    recurrent_fn: Callable[
        [torch.Tensor, torch.Tensor],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ],
    num_simulations: int,
    num_actions: int,
    *,
    legal_mask: Optional[torch.Tensor] = None,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
    """Runs batched MuZero MCTS search matching mctx semantics."""
    device = root_embeddings.device
    batch_size = root_embeddings.shape[0]
    batch_range = torch.arange(batch_size, device=device)

    # 1. Apply Dirichlet noise to root logits if enabled (Masking is handled inside init_mcts_tree)
    curr_root_logits = root_logits.clone()
    if dirichlet_epsilon > 0.0:
        curr_root_logits = add_dirichlet_noise(
            curr_root_logits,
            dirichlet_epsilon,
            dirichlet_alpha,
            mask=legal_mask,
        )

    # 2. Initialize Tree State
    tree = init_mcts_tree(
        root_embeddings=root_embeddings,
        root_logits=curr_root_logits,
        root_value=root_value,
        num_simulations=num_simulations,
        num_actions=num_actions,
        legal_mask=legal_mask,
    )

    # 3. Simulation Loop
    for _ in range(num_simulations):
        # A. Selection Phase
        leaf_parents, leaf_actions, expansion_mask, trajectory = select_leaf(
            tree,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            qtransform=qtransform,
        )

        if not trajectory:
            break

        # B. Model Dynamics Step
        parent_embeddings = tree["embeddings"][batch_range, leaf_parents]
        (
            prior_logits,
            value,
            reward,
            discount,
            next_embeddings,
        ) = recurrent_fn(leaf_actions, parent_embeddings)

        # C. Expansion Phase
        expand_node_(
            tree=tree,
            parent_nodes=leaf_parents,
            actions_taken=leaf_actions,
            policy_logits=prior_logits,
            value=value,
            rewards=reward,
            discounts=discount,
            next_embeddings=next_embeddings,
            legal_mask=None,
            masks=expansion_mask,
        )

        # D. Backpropagation Phase
        backpropagate_(
            tree=tree,
            trajectory=trajectory,
            leaf_value=value,
        )

    # 4. Extract Visit Policy and Select Actions
    root_visits = tree["children_visits"][:, 0]  # [B, A]
    action_probs = get_mcts_visit_policy(root_visits, temperature=temperature)

    if temperature == 0.0:
        selected_action = action_probs.argmax(dim=-1)
    else:
        selected_action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

    return selected_action, action_probs, tree
