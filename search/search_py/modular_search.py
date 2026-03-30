from abc import ABC
from typing import Any, Dict, Callable, List, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from modules.utils import support_to_scalar
from modules.models.inference_output import (
    InferenceOutput,
    batch_recurrent_state,
    unbatch_recurrent_state,
)
from search.search_selectors import (
    SelectionStrategy,
    TopScoreSelection,
    SamplingSelection,
)
from search.backpropogation import Backpropagator, AverageDiscountedReturnBackpropagator
from search.initial_searchsets import SearchSet, SelectAll, SelectTopK
from search.nodes import ChanceNode, DecisionNode
from search.min_max_stats import MinMaxStats
from search.prior_injectors import (
    PriorInjector,
    ActionTargetInjector,
    DirichletInjector,
    GumbelInjector,
)
from search.root_policies import (
    RootPolicyStrategy,
    CompletedQValuesRootPolicy,
    VisitFrequencyPolicy,
)
from utils.utils import get_legal_moves
from search.pruners import PruningMethod, NoPruning, SequentialHalvingPruning
from search.scoring_methods import (
    GumbelScoring,
    LeastVisitedScoring,
    UCBScoring,
    DeterministicChanceScoring,
)
from modules.models.agent_network import AgentNetwork
from .utils import _safe_log_prob


class ModularSearch:
    def __init__(self, config, device: torch.device, num_actions: int):
        """Initialise the Python MCTS backend.

        All strategy objects are derived from ``config`` automatically.
        The active strategy set is selected by ``config.gumbel``.
        """
        self.config = config
        self.device = device
        self.num_actions = num_actions

        if config.gumbel:
            self.root_selection_strategy: SelectionStrategy = TopScoreSelection(
                LeastVisitedScoring()
            )
            self.decision_selection_strategy: SelectionStrategy = TopScoreSelection(
                GumbelScoring(config)
            )
            self.chance_selection_strategy: SelectionStrategy = TopScoreSelection(
                DeterministicChanceScoring()
            )
            self.root_target_policy: RootPolicyStrategy = CompletedQValuesRootPolicy(
                config, device, num_actions
            )
            self.root_exploratory_policy: RootPolicyStrategy = VisitFrequencyPolicy(
                config, device, num_actions
            )
            self.prior_injectors: List[PriorInjector] = [
                ActionTargetInjector(),
                GumbelInjector(),
            ]
            self.root_searchset: SearchSet = SelectTopK()
            self.internal_searchset: SearchSet = SelectAll()
            self.pruning_method: PruningMethod = SequentialHalvingPruning()
            self.internal_pruning_method: PruningMethod = NoPruning()
            self.backpropagator: Backpropagator = (
                AverageDiscountedReturnBackpropagator()
            )
        else:
            _bootstrap = config.bootstrap_method
            self.root_selection_strategy: SelectionStrategy = TopScoreSelection(
                UCBScoring(bootstrap_method=_bootstrap)
            )
            self.decision_selection_strategy: SelectionStrategy = TopScoreSelection(
                UCBScoring(bootstrap_method=_bootstrap)
            )
            self.chance_selection_strategy: SelectionStrategy = TopScoreSelection(
                DeterministicChanceScoring()
            )

            from search.search_py.root_policies import (
                BestActionRootPolicy,
                VisitFrequencyPolicy,
            )

            if self.config.policy_extraction == "minimax":
                self.root_target_policy: RootPolicyStrategy = BestActionRootPolicy(
                    config, device, num_actions
                )
                self.root_exploratory_policy: RootPolicyStrategy = BestActionRootPolicy(
                    config, device, num_actions
                )
            else:
                self.root_target_policy: RootPolicyStrategy = VisitFrequencyPolicy(
                    config, device, num_actions
                )
                self.root_exploratory_policy: RootPolicyStrategy = VisitFrequencyPolicy(
                    config, device, num_actions
                )

            self.prior_injectors: List[PriorInjector] = [
                ActionTargetInjector(),
                DirichletInjector(),
            ]
            self.root_searchset: SearchSet = SelectAll()
            self.internal_searchset: SearchSet = SelectAll()
            self.pruning_method: PruningMethod = NoPruning()
            self.internal_pruning_method: PruningMethod = NoPruning()

            from search.search_py.backpropogation import (
                MinimaxBackpropagator,
                AverageDiscountedReturnBackpropagator,
            )

            if (
                hasattr(self.config, "backprop_method")
                and self.config.backprop_method == "minimax"
            ):
                self.backpropagator: Backpropagator = MinimaxBackpropagator()
            else:
                self.backpropagator: Backpropagator = (
                    AverageDiscountedReturnBackpropagator()
                )

    def _dist_for_batch_index(self, policy_dist, index: int):
        # Optimization: If the distribution is already a single-batch distribution, return it.
        # However, for batched distributions, we need to slice the logits/probs.
        logits = policy_dist.logits
        if logits is not None:
            if logits.dim() <= 1:
                batch_logits = logits
            else:
                batch_logits = logits[index]

            # Avoid .cpu() if already on CPU
            if batch_logits.device.type != "cpu":
                batch_logits = batch_logits.cpu()

            return Categorical(logits=batch_logits)

        probs = policy_dist.probs
        if probs is not None:
            if probs.dim() <= 1:
                batch_probs = probs
            else:
                batch_probs = probs[index]

            if batch_probs.device.type != "cpu":
                batch_probs = batch_probs.cpu()

            return Categorical(probs=batch_probs)

        return None

    @torch.inference_mode()
    def run(
        self,
        observation: Any,
        info: Dict[str, Any],
        agent_network: AgentNetwork,
        trajectory_action=None,
        exploration: bool = True,
    ):
        assert "player" in info, "info must contain 'player'. Got keys: " + str(
            list(info.keys())
        )
        player_raw = info["player"]
        to_play: int = int(
            player_raw.item() if torch.is_tensor(player_raw) else player_raw
        )
        self._set_node_configs()
        root = DecisionNode(0.0)

        # Delegate root expansion to the strategy
        # 1. Inference
        assert not root.expanded()

        # Root inference expects a batch dimension [1, ...]
        # We unsqueeze here since the Actor/Selector might have squeezed it for Tier-1 efficiency.
        if torch.is_tensor(observation) and observation.dim() == len(agent_network.input_shape):
            observation = observation.unsqueeze(0)
            
        outputs: InferenceOutput = agent_network.obs_inference(observation)


        val_raw = outputs.value
        root_policy_dist = outputs.policy
        policy_logits = root_policy_dist.logits
        if policy_logits is None:
            policy_probs = root_policy_dist.probs
            if policy_probs is None:
                raise ValueError(
                    "Search requires a policy distribution with logits/probs."
                )
            policy_logits = self._safe_log_prob(policy_probs)
        if policy_logits.dim() == 1:
            policy_logits = policy_logits.unsqueeze(0)
        network_state = outputs.recurrent_state

        # 3. Legal Moves
        # TODO: MOVE THE MASKING INTO THE ACTOR
        legal_moves = get_legal_moves(info)
        # print("legal smoves", legal_moves)
        if legal_moves is None:
            legal_moves = [list(range(self.num_actions))]

        # Mask in logit space so illegal actions stay exactly at -inf/zero prob.
        masked_logits = self.root_selection_strategy.mask_actions(
            policy_logits, legal_moves, device=self.device
        )

        # No terminal state (no legal moves)
        assert not (masked_logits == -float("inf")).all(dim=-1).any()

        masked_policy = torch.softmax(masked_logits, dim=-1)

        legal_moves = legal_moves[0]
        policy = masked_policy[0].cpu()  # ensure CPU for manipulation
        network_policy = policy.clone()
        network_policy_dist = Categorical(logits=masked_logits[0].cpu())
        policy_dist_for_injectors = network_policy_dist

        # Initialize reward hidden states (empty for root)
        # However, if using ValuePrefix, we might need them initialized.
        # But separate reward head has its own hidden state.
        # For initial inference, we start fresh.
        # State is already in hidden_state (opaque network_state)

        # 2. Value Processing
        # Value is already expected value from InferenceOutput
        v_pi_scalar = float(val_raw)

        # 5. Apply Prior Injectors (Stackable)
        for injector in self.prior_injectors:
            policy = injector.inject(
                policy,
                legal_moves,
                self.config,
                trajectory_action,
                policy_dist=policy_dist_for_injectors,
                exploration=exploration,
            )
            # Keep distribution/logits in sync for subsequent injectors.
            policy_dist_for_injectors = Categorical(probs=policy)

        # 6. Select Actions
        selection_count = self.config.gumbel_m
        selected_actions = self.root_searchset.create_initial_searchset(
            policy, legal_moves, selection_count, trajectory_action
        )

        # NOTE: Old MuZero parity testing only. The legacy search only seeded the
        # root visit count here and did not preload the root value accumulator.
        root.visits += 1

        # 7. Expand Root
        root.expand(
            allowed_actions=selected_actions,
            to_play=to_play,
            priors=policy,
            network_policy=network_policy,
            network_policy_dist=network_policy_dist,
            network_state=network_state,
            reward=0.0,
            value=v_pi_scalar,
        )

        min_max_stats = MinMaxStats(
            self.config.known_bounds,
            epsilon=self.config.min_max_epsilon,
        )

        # Initialize pruning state (e.g. Sequential Halving budget)
        # pruning_state = self.pruning_method.initialize(root, self.config)
        pruning_context = {
            "root": self.pruning_method.initialize(root, self.config),
            "internal": {},  # Map node -> state
        }
        # --- Main Simulation Loop ---
        search_batch_size = self.config.search_batch_size
        if search_batch_size > 0:
            num_batches = math.ceil(self.config.num_simulations / search_batch_size)
            for i in range(num_batches):
                self._run_batched_simulations(
                    root,
                    min_max_stats,
                    agent_network,
                    search_batch_size,
                    current_sim_idx=i * search_batch_size,
                    pruning_context=pruning_context,
                )
        else:
            for i in range(self.config.num_simulations):
                # Pruning method determines which actions are allowed for this simulation step
                # allowed_actions, pruning_state = self.pruning_method.step(
                #     root, pruning_state, self.config, min_max_stats, i
                # )

                self._run_single_simulation(
                    root,
                    min_max_stats,
                    agent_network,
                    # pruned_searchset=allowed_actions,
                    current_sim_idx=i,
                    pruning_context=pruning_context,
                )

        target_policy = self.root_target_policy.get_policy(root, min_max_stats)
        exploratory_policy = self.root_exploratory_policy.get_policy(
            root, min_max_stats
        )

        # --- Best Action Selection (Paper Alg. 2) ---
        # For Gumbel MuZero the move played must maximise the Gumbel score
        # A_{n+1} = g + σ(completedQ), where g is the injected Gumbel noise
        # already baked into root.child_priors by the GumbelInjector.
        # Non-Gumbel searches use argmax of the clean target policy.
        if self.config.gumbel:
            from search.search_py.utils import (
                get_completed_q,
                calculate_gumbel_sigma,
            )

            completedQ = get_completed_q(root, min_max_stats)
            sigma = calculate_gumbel_sigma(
                self.config.gumbel_cvisit,
                self.config.gumbel_cscale,
                root,
                completedQ,
            )
            # Recover g from the Gumbel-injected child_priors in log-space.
            # child_priors == 0 for masked actions → map to -inf to keep them excluded.
            noisy_logits = torch.where(
                root.child_priors > 0,
                torch.log(root.child_priors),
                torch.full_like(root.child_priors, -float("inf")),
            )
            best_action = torch.argmax(noisy_logits + sigma).item()
        else:
            best_action = torch.argmax(target_policy).item()

        return (
            root.value(),
            exploratory_policy,
            target_policy,
            best_action,
            {
                "network_policy": network_policy,
                "network_value": v_pi_scalar,
                "search_policy": target_policy,
                "search_value": root.value(),
            },
        )

    @torch.inference_mode()
    def run_vectorized(
        self,
        batched_obs: Any,
        batch_info: Dict[str, Any],
        agent_network: AgentNetwork,
        trajectory_actions=None,
        exploration: bool = True,
    ):
        # 0. Handle standardized Dict info
        if "infos_list" in batch_info:
            batched_info = batch_info["infos_list"]
        else:
            # Assume it's already a list or a dict-of-tensors
            batched_info = batch_info

        # If it's a dict of tensors (AO-style), we might need to unbatch it for modular_search's
        # Python-heavy loop (modular_search is not as vectorized as aos_search).
        # We'll just assume it's a list for now or unbatch it if it's a dict.
        if isinstance(batched_info, dict):
            B = (
                batched_obs.shape[0]
                if torch.is_tensor(batched_obs)
                else len(batched_obs)
            )
            # Standardize back to a list of dicts for Python MCTS loops
            # This is slow but modular_search is slow anyway.
            new_batched_info = []
            for i in range(B):
                item_dict = {}
                for k, v in batched_info.items():
                    # If v is a collection that matches the batch size, index it
                    if hasattr(v, "__getitem__") and not isinstance(v, (str, bytes, dict)):
                        # Some collections like tensors or arrays
                        try:
                            if len(v) == B:
                                val = v[i]
                            else:
                                val = v
                        except:
                            val = v
                    else:
                        val = v
                    
                    # Convert single-element tensors to scalars for Python loops
                    if torch.is_tensor(val) and val.numel() == 1:
                        val = val.item()
                    item_dict[k] = val
                new_batched_info.append(item_dict)
            batched_info = new_batched_info

        assert all(
            "player" in i for i in batched_info
        ), "Every info dict in batched_info must contain 'player'."
        batched_to_play: List[int] = [i["player"] for i in batched_info]
        self._set_node_configs()

        B = (
            batched_obs.shape[0]
            if isinstance(batched_obs, torch.Tensor)
            else len(batched_obs)
        )
        roots = [DecisionNode(0.0) for _ in range(B)]

        # 1. Initial Inference
        outputs: InferenceOutput = agent_network.obs_inference(batched_obs)

        val_raw = outputs.value
        root_policy_dist = outputs.policy
        policy_logits = root_policy_dist.logits
        if policy_logits is None:
            policy_probs = root_policy_dist.probs
            if policy_probs is None:
                raise ValueError(
                    "Search requires a policy distribution with logits/probs."
                )
            policy_logits = self._safe_log_prob(policy_probs)

        if trajectory_actions is None:
            trajectory_actions = [None] * B

        unbatched_states = unbatch_recurrent_state(outputs.recurrent_state)

        # Legal Moves
        legal_moves_batch = get_legal_moves(batched_info)
        if legal_moves_batch is None:
            legal_moves_batch = [list(range(self.num_actions))] * B


        # 2. Expand all B roots
        min_max_stats_list = []
        pruning_contexts_list = []
        network_policies = []

        for b in range(B):
            legal_moves = legal_moves_batch[b]

            # Mask in logit space
            masked_logits = self.root_selection_strategy.mask_actions(
                policy_logits[b : b + 1], [legal_moves], device=self.device
            )

            # No terminal state (no legal moves)
            assert not (masked_logits[0] == -float("inf")).all()

            masked_policy = torch.softmax(masked_logits, dim=-1)

            policy = masked_policy[0].cpu()
            network_policy = policy.clone()
            network_policies.append(network_policy)

            network_policy_dist = Categorical(logits=masked_logits[0].cpu())
            policy_dist_for_injectors = network_policy_dist

            v_pi_scalar = float(val_raw[b])

            # Apply Prior Injectors
            for injector in self.prior_injectors:
                policy = injector.inject(
                    policy,
                    legal_moves,
                    self.config,
                    trajectory_actions[b],
                    policy_dist=policy_dist_for_injectors,
                    exploration=exploration,
                )
                policy_dist_for_injectors = Categorical(probs=policy)

            selection_count = self.config.gumbel_m
            selected_actions = self.root_searchset.create_initial_searchset(
                policy, legal_moves, selection_count, trajectory_actions[b]
            )

            root = roots[b]
            # NOTE: Old MuZero parity testing only. Match the legacy root init
            # path and avoid preloading the root value accumulator.
            root.visits += 1
            root.expand(
                allowed_actions=selected_actions,
                to_play=batched_to_play[b],
                priors=policy,
                network_policy=network_policy,
                network_policy_dist=network_policy_dist,
                network_state=unbatched_states[b],
                reward=0.0,
                value=v_pi_scalar,
            )

            min_max_stats = MinMaxStats(
                self.config.known_bounds,
                epsilon=self.config.min_max_epsilon,
            )
            min_max_stats_list.append(min_max_stats)

            pruning_context = {
                "root": self.pruning_method.initialize(root, self.config),
                "internal": {},
            }
            pruning_contexts_list.append(pruning_context)

        # 3. Main Simulation Loop
        search_batch_size = max(1, self.config.search_batch_size)
        num_batches = math.ceil(self.config.num_simulations / search_batch_size)

        for i in range(num_batches):
            self._run_batched_vectorized_simulations(
                roots,
                min_max_stats_list,
                agent_network,
                search_batch_size,
                current_sim_idx=i * search_batch_size,
                pruning_contexts_list=pruning_contexts_list,
            )

        # 4. Extract Policies and Action
        root_values = []
        exploratory_policies = []
        target_policies = []
        best_actions = []
        search_metadata_list = []

        for b in range(B):
            root = roots[b]
            min_max_stats = min_max_stats_list[b]

            target_policy = self.root_target_policy.get_policy(root, min_max_stats)
            exploratory_policy = self.root_exploratory_policy.get_policy(
                root, min_max_stats
            )

            root_values.append(root.value())
            exploratory_policies.append(exploratory_policy)
            target_policies.append(target_policy)
            # Gumbel MuZero: play argmax(g + σ) — Paper Alg. 2.
            # Non-Gumbel: play argmax of the clean target policy.
            if self.config.gumbel:
                from search.search_py.utils import (
                    get_completed_q,
                    calculate_gumbel_sigma,
                )

                completedQ_b = get_completed_q(root, min_max_stats)
                sigma_b = calculate_gumbel_sigma(
                    self.config.gumbel_cvisit,
                    self.config.gumbel_cscale,
                    root,
                    completedQ_b,
                )
                noisy_logits_b = torch.where(
                    root.child_priors > 0,
                    torch.log(root.child_priors),
                    torch.full_like(root.child_priors, -float("inf")),
                )
                best_action_b = torch.argmax(noisy_logits_b + sigma_b)
            else:
                best_action_b = torch.argmax(target_policy)
            best_actions.append(best_action_b)
            search_metadata_list.append(
                {
                    "network_policy": network_policies[b],
                    "network_value": float(val_raw[b]),
                    "search_policy": target_policy,
                    "search_value": root.value(),
                }
            )

        return (
            root_values,
            exploratory_policies,
            target_policies,
            best_actions,
            search_metadata_list,
        )

    def _set_node_configs(self):
        ChanceNode.bootstrap_method = self.config.bootstrap_method
        ChanceNode.discount = self.config.discount_factor
        ChanceNode.discount = self.config.discount_factor
        DecisionNode.bootstrap_method = self.config.bootstrap_method
        DecisionNode.discount = self.config.discount_factor
        DecisionNode.pb_c_init = self.config.pb_c_init
        DecisionNode.pb_c_base = self.config.pb_c_base
        DecisionNode.gumbel = self.config.gumbel
        DecisionNode.cvisit = self.config.gumbel_cvisit
        DecisionNode.cscale = self.config.gumbel_cscale
        DecisionNode.stochastic = self.config.stochastic

    def _run_single_simulation(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        agent_network,
        current_sim_idx=0,
        pruning_context=None,
    ):
        node = root
        search_path = [node]
        action_path = []
        to_play = root.to_play
        # old_to_play = to_play
        # GO UNTIL A LEAF NODE IS REACHED
        # while node.expanded():
        #     action, node = node.select_child(
        #         min_max_stats=min_max_stats,
        #         pruned_searchset=pruned_searchset,
        #     )
        #     # old_to_play = (old_to_play + 1) % self.config.game.num_players
        #     search_path.append(node)
        #     horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len
        # ---------------------------------------------------------------------
        # 1. SELECTION PHASE
        # ---------------------------------------------------------------------
        # We descend until we hit a leaf DecisionNode OR a ChanceNode that needs a new code.
        node, search_path, action_path, action_or_code = self._select_child(
            node,
            search_path,
            action_path,
            min_max_stats,
            current_sim_idx,
            pruning_context,
        )
        if node is None:
            return

        parent = search_path[-2]
        value, to_play = self._expand_node(node, parent, action_or_code, agent_network)

        self._backpropagate(search_path, action_path, value, to_play, min_max_stats)

    def _select_child(
        self,
        node,
        search_path,
        action_path,
        min_max_stats,
        current_sim_idx,
        pruning_context,
    ):
        while True:
            if not node.expanded():
                break

            # If we've reached the maximum search depth, stop descending.
            # search_path already contains the root at index 0, so len(search_path) - 1 is the current depth.
            if (len(search_path) - 1) >= self.config.max_search_depth:
                break
                # Decision -> Select Action -> ChanceNode

            # Use root strategy if parent is None, otherwise use internal strategy
            if node.parent is None:
                pruned_searchset, next_state = self.pruning_method.step(
                    node,
                    pruning_context["root"],
                    self.config,
                    min_max_stats,
                    current_sim_idx,
                )
                pruning_context["root"] = next_state
                # TODO: EARLY STOPPING CLASSES
                if pruned_searchset is not None and len(pruned_searchset) == 0:
                    return None, search_path, action_path, None

                action_or_code, node = self.root_selection_strategy.select_child(
                    node,
                    pruned_searchset=pruned_searchset,
                    min_max_stats=min_max_stats,
                )
            else:
                if node.is_decision:
                    if node not in pruning_context["internal"]:
                        pruning_context["internal"][node] = (
                            self.internal_pruning_method.initialize(node, self.config)
                        )

                    pruned_searchset, next_state = self.internal_pruning_method.step(
                        node,
                        pruning_context["internal"][node],
                        self.config,
                        min_max_stats,
                        current_sim_idx,
                    )

                    # TODO: EARLY STOPPING CLASSES
                    if pruned_searchset is not None and len(pruned_searchset) == 0:
                        return None, search_path, action_path, None

                    pruning_context["internal"][node] = next_state
                    action_or_code, node = (
                        self.decision_selection_strategy.select_child(
                            node,
                            pruned_searchset=pruned_searchset,
                            min_max_stats=min_max_stats,
                        )
                    )
                else:
                    action_or_code, node = self.chance_selection_strategy.select_child(
                        node,
                        # pruned_searchset=pruned_searchset,
                        min_max_stats=min_max_stats,
                    )

            search_path.append(node)
            action_path.append(action_or_code)

        return node, search_path, action_path, action_or_code

    def _expand_node(self, node, parent, action_or_code, agent_network):
        if node.is_decision:
            if parent.is_decision:
                outputs: InferenceOutput = agent_network.hidden_state_inference(
                    parent.recurrent_state,
                    torch.as_tensor([action_or_code], device=self.device),
                )

                reward = outputs.reward
                network_state = outputs.recurrent_state
                value = outputs.value
                policy = outputs.policy.probs
                if policy is None:
                    logits = outputs.policy.logits
                    if logits is None:
                        raise ValueError(
                            "hidden_state_inference policy must expose probs/logits."
                        )
                    policy = torch.softmax(logits, dim=-1)
                if policy.dim() == 1:
                    policy = policy.unsqueeze(0)
                node_policy_dist = self._dist_for_batch_index(outputs.policy, 0)

                to_play = outputs.to_play

                reward = reward.item()
                value = value.item()
                to_play = int(to_play.item())

                actions_to_expand = self.internal_searchset.create_initial_searchset(
                    policy[0],
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy[0],
                    network_policy=policy[0],
                    network_policy_dist=node_policy_dist,
                    network_state=network_state,
                    reward=reward,
                    value=value,
                )
            elif parent.is_chance:
                # Prepare scalar code as one-hot for inference
                action_t = torch.tensor(action_or_code, device=self.device)
                num_codes = parent.child_priors.shape[0]
                action_t = action_t.long()
                one_hot_code = F.one_hot(action_t, num_classes=num_codes)

                outputs: InferenceOutput = agent_network.hidden_state_inference(
                    parent.recurrent_state,
                    one_hot_code.unsqueeze(0).float(),
                )

                reward = float(outputs.reward)
                network_state = outputs.recurrent_state
                value = float(outputs.value)
                policy = outputs.policy.probs
                if policy is None:
                    logits = outputs.policy.logits
                    if logits is None:
                        raise ValueError(
                            "hidden_state_inference policy must expose probs/logits."
                        )
                    policy = torch.softmax(logits, dim=-1)
                if policy.dim() == 1:
                    policy = policy.unsqueeze(0)
                node_policy_dist = self._dist_for_batch_index(outputs.policy, 0)

                to_play = int(outputs.to_play)

                actions_to_expand = self.internal_searchset.create_initial_searchset(
                    policy[0],
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy[0],
                    network_policy=policy[0],
                    network_policy_dist=node_policy_dist,
                    network_state=network_state,
                    reward=reward,
                    value=value,
                )
        elif node.is_chance:
            # CASE B: Stochastic Expansion (The Core Change)
            # We are at (State, Action). We need to:
            # 1. Get Afterstate Value & Code Priors (Expand ChanceNode)
            # 2. Sample a Code
            # 3. Get Next State & Reward (Create DecisionNode)
            outputs: InferenceOutput = agent_network.afterstate_inference(
                parent.recurrent_state,
                torch.as_tensor(
                    [action_or_code],
                    device=self.device,
                    dtype=torch.float,
                ),
            )

            network_state = outputs.recurrent_state
            value = float(outputs.value)
            code_probs = outputs.policy.probs

            # Expand the Chance Node with these priors
            node.expand(
                to_play=parent.to_play,
                network_state=network_state,
                network_value=value,
                code_probs=code_probs[0],
            )
            to_play = parent.to_play

        return value, to_play

    def _backpropagate(self, search_path, action_path, value, to_play, min_max_stats):
        self.backpropagator.backpropagate(
            search_path, action_path, value, to_play, min_max_stats, self.config
        )

    def _run_batched_simulations(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        agent_network,
        batch_size,
        current_sim_idx=0,
        pruning_context=None,
    ):
        use_virtual_mean = self.config.use_virtual_mean
        virtual_loss = self.config.virtual_loss

        sim_data = []

        # 1. Selection Phase
        for b in range(batch_size):
            node = root
            search_path = [node]
            action_path = []
            path_virtual_values = (
                []
            )  # Track virtual values for this path if using Virtual Mean
            horizon_index = 0

            action_or_code = None

            action_or_code = None

            while True:
                if not node.expanded():
                    break

                if (
                    self.config.max_search_depth is not None
                    and (len(search_path) - 1) >= self.config.max_search_depth
                ):
                    break

                parent_node = node

                if node.parent is None:
                    # Root
                    pruned_searchset, next_state = self.pruning_method.step(
                        node,
                        pruning_context["root"],
                        self.config,
                        min_max_stats,
                        current_sim_idx + b,
                    )
                    pruning_context["root"] = next_state
                    if pruned_searchset is not None and len(pruned_searchset) == 0:
                        # Revert virtual update for this failed path
                        # existing nodes in search_path (including root) have been updated
                        if use_virtual_mean:
                            for n, v in zip(search_path, path_virtual_values):
                                n.visits -= 1
                                n.value_sum -= v
                                n._v_mix = None
                        else:
                            # Revert standard VL
                            for n in search_path:
                                n.visits -= 1
                                n.value_sum += virtual_loss
                                n._v_mix = None

                        for i in range(len(action_path)):
                            p_node = search_path[i]
                            act = int(action_path[i])
                            p_node.child_visits[act] -= 1

                        node = None
                        break

                    action_or_code, node = self.root_selection_strategy.select_child(
                        node,
                        pruned_searchset=pruned_searchset,
                        min_max_stats=min_max_stats,
                    )
                else:
                    if node.is_decision:
                        if node not in pruning_context["internal"]:
                            pruning_context["internal"][node] = (
                                self.internal_pruning_method.initialize(
                                    node, self.config
                                )
                            )

                        pruned_searchset, next_state = (
                            self.internal_pruning_method.step(
                                node,
                                pruning_context["internal"][node],
                                self.config,
                                min_max_stats,
                                current_sim_idx + b,
                            )
                        )
                        pruning_context["internal"][node] = next_state

                        if pruned_searchset is not None and len(pruned_searchset) == 0:
                            if use_virtual_mean:
                                for n, v in zip(search_path, path_virtual_values):
                                    n.visits -= 1
                                    n.value_sum -= v
                                    n._v_mix = None
                            else:
                                for n in search_path:
                                    n.visits -= 1
                                    n.value_sum += virtual_loss
                                    n._v_mix = None

                            # Fix: Revert child_visits
                            for i in range(len(action_path)):
                                p_node = search_path[i]
                                act = int(action_path[i])
                                p_node.child_visits[act] -= 1

                            node = None
                            break

                        action_or_code, node = (
                            self.decision_selection_strategy.select_child(
                                node,
                                pruned_searchset=pruned_searchset,
                                min_max_stats=min_max_stats,
                            )
                        )
                    elif node.is_chance:
                        action_or_code, node = (
                            self.chance_selection_strategy.select_child(
                                node,
                                min_max_stats=min_max_stats,
                            )
                        )

                # parent_node = search_path[-1] # Redundant with loop logic

                # Apply virtual update to the PARENT (search_path[-1])
                # We do this after successful selection.
                parent_node = search_path[-1]

                # CRITICAL FIX: Sync child_visits tensor for vectorized selection
                # This ensures subsequent batch items see the updated visit count
                act_idx = int(action_or_code)

                parent_node.child_visits[act_idx] += 1

                if use_virtual_mean:
                    v_val = parent_node.value()
                    parent_node.visits += 1
                    parent_node.value_sum += v_val
                    # Invalidate v_mix because parent_node.value() changed
                    parent_node._v_mix = None

                    path_virtual_values.append(
                        v_val
                    )  # Note: path_virtual_values will match search_path indices
                else:
                    parent_node.visits += 1
                    parent_node.value_sum -= virtual_loss
                    # Invalidate v_mix
                    parent_node._v_mix = None

                search_path.append(node)
                action_path.append(action_or_code)

            if node is None:
                continue

            # Leaf Node Update (since loop only updates parents)
            if use_virtual_mean:
                v_val = node.value()  # Bootstrap
                node.visits += 1
                node.value_sum += v_val
                node._v_mix = None
                path_virtual_values.append(v_val)
            else:
                node.visits += 1
                node.value_sum -= virtual_loss
                node._v_mix = None

            # Leaf already updated in the loop (added to search_path and updated)
            # Check: Loop breaks when `not node.expanded()`.
            # Inside loop: we select child `node`. Then update `node`. Then append to search_path.
            # Then check `if not node.expanded(): break`.
            # So `node` (the leaf) IS in search_path and HAS been updated.
            # In original code:
            #  parent_node updated in loop.
            #  search_path appended.
            #  Leaf updated AFTER loop.
            #  Wait, let's check original code carefully.
            #  Original:
            #    while True:
            #      if not node.expanded(): break
            #      ... select ...
            #      parent_node.visits += 1... (Update PARENT)
            #      search_path.append(node) (Child)
            #    node.visits += 1... (Update LEAF)
            #
            #  My new code:
            #    Update ROOT (start of loop).
            #    while True:
            #       if not node.expanded(): break
            #       ... select ...
            #       Update CHILD (node).
            #       search_path.append(node).
            #
            #  Trace:
            #    Start: node=Root. Update Root. search_path=[Root].
            #    Loop 1: Root expanded? Yes.
            #       Select Child1.
            #       Update Child1.
            #       search_path=[Root, Child1].
            #       node=Child1.
            #    Loop 2: Child1 expanded? No. Break.
            #
            #  Result: Root and Child1 updated. search_path=[Root, Child1].
            #  This covers exactly everyone in search_path.
            #  Original code:
            #    Loop 1: Root expanded? Yes.
            #       Select Child1.
            #       Update parent (Root).
            #       search_path=[Root, Child1].
            #       node=Child1.
            #    Loop 2: Child1 expanded? No. Break.
            #    Update leaf (Child1).
            #
            #  Result: Root and Child1 updated.
            #  Logic is equivalent. Proceed.

            sim_data.append(
                {
                    "path": search_path,
                    "action_path": action_path,
                    "node": node,
                    "parent": search_path[-2],
                    "action": action_or_code,
                    "virtual_values": path_virtual_values if use_virtual_mean else None,
                }
            )

        # 2. Batched Inference
        recurrent_inputs = []
        afterstate_inputs = []

        for i, d in enumerate(sim_data):
            node = d["node"]
            parent = d["parent"]
            action = d["action"]

            if node.is_decision:
                state = parent.recurrent_state
                assert (
                    state is not None
                ), f"Parent node {type(parent)} at search path index {len(d['path'])-2} has network_state=None. Node type: {type(node)}"

                recurrent_inputs.append(
                    {
                        "state": state,
                        "action": action,
                        "idx": i,
                    }
                )
            elif node.is_chance:
                state = parent.recurrent_state
                assert (
                    state is not None
                ), f"Parent node {type(parent)} has network_state=None for ChanceNode expansion. Parent to_play: {parent.to_play}"
                afterstate_inputs.append(
                    {
                        "state": state,
                        "action": action,
                        "idx": i,
                    }
                )

        if recurrent_inputs:
            # 1. Batch full opaque states recursively
            full_states = [x["state"] for x in recurrent_inputs]
            batched_states = batch_recurrent_state(full_states)

            # 2. Prepare actions
            act_list = []
            for x in recurrent_inputs:
                d = sim_data[x["idx"]]
                is_chance_parent = d["parent"].is_chance
                raw_action = x["action"]
                val = torch.as_tensor(raw_action, device=self.device)

                if is_chance_parent:
                    num_codes = d["parent"].child_priors.shape[0]
                    one_hot = F.one_hot(val.long(), num_classes=num_codes)
                    act_list.append(one_hot.float().unsqueeze(0))
                else:
                    act_list.append(val.unsqueeze(0))

            actions = torch.cat(act_list, dim=0)

            # 3. Inference
            outputs: InferenceOutput = agent_network.hidden_state_inference(
                batched_states,
                actions,
            )

            # 4. Unbatch everything recursively
            unbatched_next_states = unbatch_recurrent_state(outputs.recurrent_state)

            rewards = outputs.reward
            values = outputs.value
            to_plays = outputs.to_play

            # Pre-calculate policies if needed for expansion
            policies = outputs.policy.probs
            if policies is None:
                policy_logits = outputs.policy.logits
                if policy_logits is None:
                    raise ValueError(
                        "hidden_state_inference policy must expose probs/logits."
                    )
                policies = torch.softmax(policy_logits, dim=-1)

            for local_i, x in enumerate(recurrent_inputs):
                idx = x["idx"]
                sim_data[idx]["result"] = {
                    "reward": rewards[local_i],
                    "network_state": unbatched_next_states[local_i],
                    "value": values[local_i],
                    "policy": policies[local_i],
                    "policy_dist": self._dist_for_batch_index(outputs.policy, local_i),
                    "to_play": to_plays[local_i],
                }

        if afterstate_inputs:
            # 1. Batch opaque states
            full_after_states = [x["state"] for x in afterstate_inputs]
            batched_after_states = batch_recurrent_state(full_after_states)

            actions = (
                torch.tensor(
                    [x["action"] for x in afterstate_inputs], device=self.device
                )
                .float()
                .unsqueeze(1)
            )

            outputs: InferenceOutput = agent_network.afterstate_inference(
                batched_after_states, actions
            )

            # 2. Unbatch opaque states
            unbatched_afterstates = unbatch_recurrent_state(outputs.recurrent_state)

            values = outputs.value
            code_probs_batch = outputs.policy.probs

            for local_i, x in enumerate(afterstate_inputs):
                idx = x["idx"]
                sim_data[idx]["result"] = {
                    "network_state": unbatched_afterstates[local_i],
                    "value": values[local_i],
                    "code_probs": code_probs_batch[local_i],
                }

        # 3. Expansion & Backprop

        # A. Revert Virtual Loss / Virtual Mean (Global Reversion Phase)
        # We must revert ALL virtual updates before backprop to ensure min_max_stats
        # sees clean values without penalty/mean bias from other simulations.
        for d in sim_data:
            path = d["path"]
            virtual_values = d.get("virtual_values", [])

            # Revert Path VL/VM
            if virtual_values:
                # Virtual Mean Reversion
                for node, v_val in zip(path, virtual_values):
                    node.visits -= 1
                    node.value_sum -= v_val
                    node._v_mix = None
            else:
                # Virtual Loss Reversion (Constant)
                for node in path:
                    node.visits -= 1
                    node.value_sum += virtual_loss
                    node._v_mix = None

            # CRITICAL FIX: Revert child_visits tensor
            for i in range(len(d["action_path"])):
                parent = path[i]
                action = d["action_path"][i]
                act_idx = int(action)
                parent.child_visits[act_idx] -= 1

        # B. Backpropagation Phase
        for d in sim_data:
            path = d["path"]
            node = d["node"]

            res = d.get("result")
            if not res:
                continue

            to_play_for_backprop = None

            if node.is_decision:
                reward = res["reward"]
                value = res["value"]

                # Check for distributional support range
                if self.config.support_range is not None:
                    if reward.numel() > 1:
                        reward = support_to_scalar(
                            reward, self.config.support_range
                        ).item()
                    else:
                        reward = reward.item()
                    if value.numel() > 1:
                        value = support_to_scalar(
                            value, self.config.support_range
                        ).item()
                    else:
                        value = value.item()
                else:
                    reward = float(reward)
                    value = float(value)

                to_play = int(res["to_play"])
                to_play_for_backprop = to_play

                policy = res["policy"]
                actions_to_expand = self.internal_searchset.create_initial_searchset(
                    policy,
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy,
                    network_policy=policy,
                    network_policy_dist=res.get("policy_dist"),
                    network_state=res["network_state"],
                    reward=reward,
                    value=value,
                )

            elif node.is_chance:
                value = res["value"]
                if self.config.support_range:
                    if value.numel() > 1:
                        value = support_to_scalar(
                            value, self.config.support_range
                        ).item()
                    else:
                        value = value.item()
                else:
                    value = float(value)

                to_play_for_backprop = d["parent"].to_play

                node.expand(
                    to_play=d["parent"].to_play,
                    network_state=res["network_state"],
                    network_value=value,
                    code_probs=res["code_probs"],
                )

            self.backpropagator.backpropagate(
                path,
                d["action_path"],
                value,
                to_play_for_backprop,
                min_max_stats,
                self.config,
            )

    def _run_batched_vectorized_simulations(
        self,
        roots: List[DecisionNode],
        min_max_stats_list: List[MinMaxStats],
        agent_network,
        search_batch_size,
        current_sim_idx=0,
        pruning_contexts_list=None,
    ):
        use_virtual_mean = self.config.use_virtual_mean
        virtual_loss = self.config.virtual_loss

        sim_data = []
        B = len(roots)

        # 1. Selection Phase for all B trees
        for b in range(B):
            root = roots[b]
            min_max_stats = min_max_stats_list[b]
            pruning_context = pruning_contexts_list[b]

            for path_idx in range(search_batch_size):
                node = root
                search_path = [node]
                action_path = []
                path_virtual_values = []
                action_or_code = None

                while True:
                    if not node.expanded():
                        break

                    parent_node = node
                    if node.parent is None:
                        # Root
                        pruned_searchset, next_state = self.pruning_method.step(
                            node,
                            pruning_context["root"],
                            self.config,
                            min_max_stats,
                            current_sim_idx + path_idx,
                        )
                        pruning_context["root"] = next_state
                        if pruned_searchset is not None and len(pruned_searchset) == 0:
                            if use_virtual_mean:
                                for n, v in zip(search_path, path_virtual_values):
                                    n.visits -= 1
                                    n.value_sum -= v
                                    n._v_mix = None
                            else:
                                for n in search_path:
                                    n.visits -= 1
                                    n.value_sum += virtual_loss
                                    n._v_mix = None
                            for j in range(len(action_path)):
                                p_node = search_path[j]
                                act = int(action_path[j])
                                p_node.child_visits[act] -= 1
                            node = None
                            break

                        action_or_code, node = (
                            self.root_selection_strategy.select_child(
                                node,
                                pruned_searchset=pruned_searchset,
                                min_max_stats=min_max_stats,
                            )
                        )
                    else:
                        if node.is_decision:
                            if node not in pruning_context["internal"]:
                                pruning_context["internal"][node] = (
                                    self.internal_pruning_method.initialize(
                                        node, self.config
                                    )
                                )
                            pruned_searchset, next_state = (
                                self.internal_pruning_method.step(
                                    node,
                                    pruning_context["internal"][node],
                                    self.config,
                                    min_max_stats,
                                    current_sim_idx + path_idx,
                                )
                            )
                            pruning_context["internal"][node] = next_state
                            if (
                                pruned_searchset is not None
                                and len(pruned_searchset) == 0
                            ):
                                if use_virtual_mean:
                                    for n, v in zip(search_path, path_virtual_values):
                                        n.visits -= 1
                                        n.value_sum -= v
                                        n._v_mix = None
                                else:
                                    for n in search_path:
                                        n.visits -= 1
                                        n.value_sum += virtual_loss
                                        n._v_mix = None
                                for j in range(len(action_path)):
                                    p_node = search_path[j]
                                    act = int(action_path[j])
                                    p_node.child_visits[act] -= 1
                                node = None
                                break
                            action_or_code, node = (
                                self.decision_selection_strategy.select_child(
                                    node,
                                    pruned_searchset=pruned_searchset,
                                    min_max_stats=min_max_stats,
                                )
                            )
                        elif node.is_chance:
                            action_or_code, node = (
                                self.chance_selection_strategy.select_child(
                                    node,
                                    min_max_stats=min_max_stats,
                                )
                            )

                    parent_node = search_path[-1]
                    act_idx = int(action_or_code)
                    parent_node.child_visits[act_idx] += 1

                    if use_virtual_mean:
                        v_val = parent_node.value()
                        parent_node.visits += 1
                        parent_node.value_sum += v_val
                        parent_node._v_mix = None
                        path_virtual_values.append(v_val)
                    else:
                        parent_node.visits += 1
                        parent_node.value_sum -= virtual_loss
                        parent_node._v_mix = None

                    search_path.append(node)
                    action_path.append(action_or_code)

                if node is None:
                    continue

                if use_virtual_mean:
                    v_val = node.value()
                    node.visits += 1
                    node.value_sum += v_val
                    node._v_mix = None
                    path_virtual_values.append(v_val)
                else:
                    node.visits += 1
                    node.value_sum -= virtual_loss
                    node._v_mix = None

                sim_data.append(
                    {
                        "b": b,
                        "path": search_path,
                        "action_path": action_path,
                        "node": node,
                        "parent": search_path[-2],
                        "action": action_or_code,
                        "virtual_values": (
                            path_virtual_values if use_virtual_mean else None
                        ),
                    }
                )

        if not sim_data:
            return

        # 2. Batched Inference for all collected paths
        recurrent_inputs = []
        afterstate_inputs = []

        for i, d in enumerate(sim_data):
            node = d["node"]
            parent = d["parent"]
            action = d["action"]

            if node.is_decision:
                state = parent.recurrent_state
                recurrent_inputs.append({"state": state, "action": action, "idx": i})
            elif node.is_chance:
                state = parent.recurrent_state
                afterstate_inputs.append({"state": state, "action": action, "idx": i})

        if recurrent_inputs:
            full_states = [x["state"] for x in recurrent_inputs]
            if isinstance(full_states[0], dict):
                batched_states = batch_recurrent_state(full_states)
            else:
                batched_states = type(full_states[0]).batch(full_states)

            act_list = []
            for x in recurrent_inputs:
                d = sim_data[x["idx"]]
                is_chance_parent = d["parent"].is_chance
                raw_action = x["action"]
                val = torch.as_tensor(raw_action, device=self.device)
                if is_chance_parent:
                    num_codes = d["parent"].child_priors.shape[0]
                    one_hot = F.one_hot(val.long(), num_classes=num_codes)
                    act_list.append(one_hot.float().unsqueeze(0))
                else:
                    act_list.append(val.unsqueeze(0))
            actions = torch.cat(act_list, dim=0)

            outputs: InferenceOutput = agent_network.hidden_state_inference(
                batched_states, actions
            )
            unbatched_next_states = unbatch_recurrent_state(outputs.recurrent_state)

            rewards = outputs.reward
            values = outputs.value
            to_plays = outputs.to_play

            policies = outputs.policy.probs
            if policies is None:
                policies = torch.softmax(outputs.policy.logits, dim=-1)

            for local_i, x in enumerate(recurrent_inputs):
                idx = x["idx"]
                sim_data[idx]["result"] = {
                    "reward": rewards[local_i],
                    "network_state": unbatched_next_states[local_i],
                    "value": values[local_i],
                    "policy": policies[local_i],
                    "policy_dist": self._dist_for_batch_index(outputs.policy, local_i),
                    "to_play": to_plays[local_i],
                }

        if afterstate_inputs:
            full_after_states = [x["state"] for x in afterstate_inputs]
            batched_after_states = batch_recurrent_state(full_after_states)

            act_list = []
            for x in afterstate_inputs:
                raw_action = x["action"]
                val = torch.as_tensor(
                    [raw_action], device=self.device, dtype=torch.float
                )
                act_list.append(val)
            actions = torch.cat(act_list, dim=0).unsqueeze(1)

            outputs: InferenceOutput = agent_network.afterstate_inference(
                batched_after_states, actions
            )
            unbatched_next_states = unbatch_recurrent_state(outputs.recurrent_state)
            values = outputs.value
            code_probs_batch = outputs.policy.probs

            for local_i, x in enumerate(afterstate_inputs):
                idx = x["idx"]
                sim_data[idx]["result"] = {
                    "network_state": unbatched_next_states[local_i],
                    "value": values[local_i],
                    "code_probs": code_probs_batch[local_i],
                }

        # 3. Expansion & Backprop
        # A. Revert Virtual Loss / Virtual Mean
        for d in sim_data:
            path = d["path"]
            virtual_values = d.get("virtual_values", [])

            if virtual_values:
                for node, v_val in zip(path, virtual_values):
                    node.visits -= 1
                    node.value_sum -= v_val
                    node._v_mix = None
            else:
                for node in path:
                    node.visits -= 1
                    node.value_sum += virtual_loss
                    node._v_mix = None

            for i in range(len(d["action_path"])):
                parent = path[i]
                action = d["action_path"][i]
                act_idx = int(action)
                parent.child_visits[act_idx] -= 1

        # B. Backpropagation
        for d in sim_data:
            path = d["path"]
            node = d["node"]
            res = d.get("result")
            min_max_stats = min_max_stats_list[d["b"]]

            if not res:
                continue

            to_play_for_backprop = None

            if node.is_decision:
                reward = res["reward"]
                value = res["value"]

                if self.config.support_range is not None:
                    if reward.numel() > 1:
                        reward = support_to_scalar(
                            reward, self.config.support_range
                        ).item()
                    else:
                        reward = reward.item()
                    if value.numel() > 1:
                        value = support_to_scalar(
                            value, self.config.support_range
                        ).item()
                    else:
                        value = value.item()
                else:
                    reward = float(reward)
                    value = float(value)

                to_play = int(res["to_play"])
                to_play_for_backprop = to_play

                policy = res["policy"]
                actions_to_expand = self.internal_searchset.create_initial_searchset(
                    policy,
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy,
                    network_policy=policy,
                    network_policy_dist=res.get("policy_dist"),
                    network_state=res["network_state"],
                    reward=reward,
                    value=value,
                )

            elif node.is_chance:
                value = res["value"]
                if self.config.support_range is not None:
                    if value.numel() > 1:
                        value = support_to_scalar(
                            value, self.config.support_range
                        ).item()
                    else:
                        value = value.item()
                else:
                    value = float(value)

                to_play_for_backprop = d["parent"].to_play

                node.expand(
                    to_play=d["parent"].to_play,
                    network_state=res["network_state"],
                    network_value=value,
                    code_probs=(
                        res["code_probs"][0]
                        if res["code_probs"].dim() > 1
                        else res["code_probs"]
                    ),
                )

            self.backpropagator.backpropagate(
                path,
                d["action_path"],
                value,
                to_play_for_backprop,
                min_max_stats,
                self.config,
            )
