from abc import ABC
import torch
import torch.nn.functional as F
import math
from modules.utils import support_to_scalar
from search.search_selectors import SelectionStrategy
from search.backpropogation import Backpropagator
from search.initial_searchsets import SearchSet
from search.nodes import ChanceNode, DecisionNode
from search.min_max_stats import MinMaxStats
from search.prior_injectors import PriorInjector
from search.root_policies import RootPolicyStrategy
from utils.utils import action_mask, get_legal_moves
from search.pruners import PruningMethod


class SearchAlgorithm:
    def __init__(
        self,
        config,
        device,
        num_actions,
        root_selection_strategy,
        decision_selection_strategy,
        chance_selection_strategy,
        root_target_policy,
        root_exploratory_policy,
        prior_injectors,
        root_searchset,
        internal_searchset,
        pruning_method,
        internal_pruning_method,
        backpropagator,
    ):
        self.config = config
        self.device = device
        self.num_actions = num_actions

        self.root_selection_strategy: SelectionStrategy = root_selection_strategy
        self.decision_selection_strategy: SelectionStrategy = (
            decision_selection_strategy
        )
        self.chance_selection_strategy: SelectionStrategy = chance_selection_strategy
        self.root_target_policy: RootPolicyStrategy = root_target_policy
        self.root_exploratory_policy: RootPolicyStrategy = root_exploratory_policy
        self.prior_injectors: PriorInjector = prior_injectors
        self.root_searchset: SearchSet = root_searchset
        self.internal_searchset: SearchSet = internal_searchset
        self.pruning_method: PruningMethod = pruning_method
        self.internal_pruning_method: PruningMethod = internal_pruning_method
        self.backpropagator: Backpropagator = backpropagator

    def run(
        self,
        state,
        info,
        to_play,
        inference_fns,
        trajectory_action=None,
        inference_model=None,
    ):
        self._set_node_configs()
        root = DecisionNode(0.0)

        # Delegate root expansion to the strategy
        # 1. Inference
        assert not root.expanded()

        outputs = inference_fns["initial"](state, model=inference_model)
        if len(outputs) == 3:
            val_raw, policy, hidden_state = outputs
        else:
            raise ValueError("Inference function returned unexpected number of values")

        # 3. Legal Moves
        legal_moves = get_legal_moves(info)
        # print("legal smoves", legal_moves)
        if legal_moves is None:
            legal_moves = [list(range(self.num_actions))]

        # TODO: should i action mask?
        policy = action_mask(policy, legal_moves, device=self.device)

        legal_moves = legal_moves[0]
        policy = policy[0]
        policy = policy.cpu()  # ensure CPU for manipulation
        network_policy = policy.clone()

        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        # 2. Value Processing
        if self.config.support_range is not None:
            if val_raw.numel() == 1:
                v_pi_scalar = float(val_raw.item())
            else:
                v_pi = support_to_scalar(val_raw, self.config.support_range)
                v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(val_raw.item())

        # 5. Apply Prior Injectors (Stackable)
        for injector in self.prior_injectors:
            policy = injector.inject(
                policy, legal_moves, self.config, trajectory_action
            )

        # 6. Select Actions
        selection_count = self.config.gumbel_m
        selected_actions = self.root_searchset.create_initial_searchset(
            policy, legal_moves, selection_count, trajectory_action
        )

        root.visits += 1

        # 7. Expand Root
        root.expand(
            allowed_actions=selected_actions,
            to_play=to_play,
            priors=policy.to(self.device),  # Send back to device if needed by Node
            network_policy=network_policy,
            hidden_state=hidden_state,
            reward=0.0,
            value=v_pi_scalar,
            reward_h_state=reward_h_state,
            reward_c_state=reward_c_state,
            is_reset=True,
        )

        min_max_stats = MinMaxStats(
            self.config.known_bounds,
            soft_update=self.config.soft_update,
            min_max_epsilon=self.config.min_max_epsilon,
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
                    inference_fns,
                    batch_size=search_batch_size,
                    inference_model=inference_model,
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
                    inference_fns,
                    inference_model=inference_model,
                    # pruned_searchset=allowed_actions,
                    current_sim_idx=i,
                    pruning_context=pruning_context,
                )

        target_policy = self.root_target_policy.get_policy(root, min_max_stats)
        exploratory_policy = self.root_exploratory_policy.get_policy(
            root, min_max_stats
        )

        # Mask target policy if required by pruning method (e.g. for Sequential Halving)
        if self.pruning_method.mask_target_policy:
            target_policy = action_mask(
                target_policy.unsqueeze(0), [legal_moves]
            ).squeeze(0)

        assert (
            isinstance(target_policy, torch.Tensor)
            and target_policy.shape == policy.shape
        )
        return (
            root.value(),
            exploratory_policy,
            target_policy,
            # TODO: BEST ACTION SELECTION, WHERE? WHAT, HOW?
            torch.argmax(target_policy),
            {
                "network_policy": network_policy,
                "network_value": v_pi_scalar,
                "search_policy": target_policy,
                "search_value": root.value(),
            },
        )

    def _set_node_configs(self):
        ChanceNode.estimation_method = self.config.q_estimation_method
        ChanceNode.discount = self.config.discount_factor
        ChanceNode.value_prefix = self.config.value_prefix
        DecisionNode.estimation_method = self.config.q_estimation_method
        DecisionNode.discount = self.config.discount_factor
        DecisionNode.value_prefix = self.config.value_prefix
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
        inference_fns,
        inference_model=None,
        current_sim_idx=0,
        pruning_context=None,
    ):
        node = root
        search_path = [node]
        action_path = []
        to_play = root.to_play
        horizon_index = 0
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
        while True:
            if not node.expanded():
                break  # Reached a leaf state (DecisionNode)
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
                    return  # Stop this simulation

                action_or_code, node = self.root_selection_strategy.select_child(
                    node,
                    pruned_searchset=pruned_searchset,
                    min_max_stats=min_max_stats,
                )
            else:
                if isinstance(node, DecisionNode):
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
                        return  # Stop this simulation

                    pruning_context["internal"][node] = next_state
                    action_or_code, node = (
                        self.decision_selection_strategy.select_child(
                            node,
                            pruned_searchset=pruned_searchset,
                            min_max_stats=min_max_stats,
                        )
                    )
                elif isinstance(node, ChanceNode):
                    action_or_code, node = self.chance_selection_strategy.select_child(
                        node,
                        # pruned_searchset=pruned_searchset,
                        min_max_stats=min_max_stats,
                    )

            horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len
            search_path.append(node)
            action_path.append(action_or_code)

        parent = search_path[-2]
        # if to_play != old_to_play and self.training_step > 1000:
        #     print("WRONG TO PLAY", onehot_to_play)
        if isinstance(node, DecisionNode):
            if isinstance(parent, DecisionNode):
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = inference_fns["recurrent"](
                    parent.hidden_state,
                    torch.as_tensor(
                        [action_or_code], device=parent.hidden_state.device
                    ),
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
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
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

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
                    hidden_state=hidden_state,
                    reward=reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
            elif isinstance(parent, ChanceNode):
                # Prepare scalar code as one-hot for inference
                action_t = torch.tensor(action_or_code, device=parent.afterstate.device)
                num_codes = parent.child_priors.shape[0]
                action_t = action_t.long()
                one_hot_code = F.one_hot(action_t, num_classes=num_codes)

                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = inference_fns["recurrent"](
                    parent.afterstate,
                    one_hot_code.unsqueeze(0).float(),
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
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
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

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
                    hidden_state=hidden_state,
                    reward=reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
        elif isinstance(node, ChanceNode):
            # CASE B: Stochastic Expansion (The Core Change)
            # We are at (State, Action). We need to:
            # 1. Get Afterstate Value & Code Priors (Expand ChanceNode)
            # 2. Sample a Code
            # 3. Get Next State & Reward (Create DecisionNode)
            afterstate, value, code_probs = inference_fns[
                "afterstate"
            ](  # <--- YOU NEED THIS METHOD
                parent.hidden_state,
                torch.as_tensor(
                    [action_or_code],
                    device=parent.hidden_state.device,
                    dtype=torch.float,
                ),
                model=inference_model,
            )

            if self.config.support_range:
                if value.numel() > 1:
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    value = value.item()
            else:
                value = value.item()

            # Expand the Chance Node with these priors
            node.expand(
                to_play=parent.to_play,
                afterstate=afterstate,
                network_value=value,
                code_probs=code_probs[0],
                reward_h_state=parent.reward_h_state,
                reward_c_state=parent.reward_c_state,
            )

        self.backpropagator.backpropagate(
            search_path, action_path, value, to_play, min_max_stats, self.config
        )

    def _run_batched_simulations(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        inference_fns,
        batch_size,
        inference_model=None,
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

                        # Fix: Revert child_visits
                        for i in range(len(action_path)):
                            p_node = search_path[i]
                            act = action_path[i]
                            if isinstance(act, torch.Tensor):
                                act = act.item()
                            p_node.child_visits[act] -= 1

                        node = None
                        break

                    action_or_code, node = self.root_selection_strategy.select_child(
                        node,
                        pruned_searchset=pruned_searchset,
                        min_max_stats=min_max_stats,
                    )
                else:
                    if isinstance(node, DecisionNode):
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
                                act = action_path[i]
                                if isinstance(act, torch.Tensor):
                                    act = act.item()
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
                    elif isinstance(node, ChanceNode):
                        action_or_code, node = (
                            self.chance_selection_strategy.select_child(
                                node,
                                min_max_stats=min_max_stats,
                            )
                        )

                horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len

                # Apply virtual update to the PARENT (search_path[-1])
                # We do this after successful selection.
                parent_node = search_path[-1]

                # CRITICAL FIX: Sync child_visits tensor for vectorized selection
                # This ensures subsequent batch items see the updated visit count
                if isinstance(action_or_code, torch.Tensor):
                    act_idx = action_or_code.item()
                else:
                    act_idx = action_or_code

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
                    "horizon_index": horizon_index,
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

            if isinstance(node, DecisionNode):
                if isinstance(parent, DecisionNode):
                    state = parent.hidden_state
                elif isinstance(parent, ChanceNode):
                    state = parent.afterstate

                recurrent_inputs.append(
                    {
                        "state": state,
                        "action": action,
                        "rh": parent.reward_h_state,
                        "rc": parent.reward_c_state,
                        "idx": i,
                    }
                )
            elif isinstance(node, ChanceNode):
                afterstate_inputs.append(
                    {
                        "state": parent.hidden_state,
                        "action": action,
                        "idx": i,
                    }
                )

        if recurrent_inputs:
            states = torch.cat([x["state"] for x in recurrent_inputs], dim=0)

            act_list = []
            for x in recurrent_inputs:
                d = sim_data[x["idx"]]
                is_chance_parent = isinstance(d["parent"], ChanceNode)
                raw_action = x["action"]
                if isinstance(raw_action, torch.Tensor):
                    val = raw_action.to(self.device).detach().clone()
                else:
                    val = torch.as_tensor(raw_action, device=self.device)

                if is_chance_parent:
                    # Convert scalar code to one-hot
                    num_codes = d["parent"].child_priors.shape[0]
                    val = val.long()  # one_hot expects long
                    one_hot = F.one_hot(val, num_classes=num_codes)
                    act_list.append(one_hot.float().unsqueeze(0))
                else:
                    act_list.append(val.unsqueeze(0))

            actions = torch.cat(act_list, dim=0)

            rhs = torch.cat([x["rh"] for x in recurrent_inputs], dim=0)
            rcs = torch.cat([x["rc"] for x in recurrent_inputs], dim=0)

            (
                rewards,
                hidden_states,
                values,
                policies,
                to_plays,
                rh_news,
                rc_news,
            ) = inference_fns["recurrent"](
                states, actions, rhs, rcs, model=inference_model
            )

            for local_i, x in enumerate(recurrent_inputs):
                idx = x["idx"]
                sim_data[idx]["result"] = {
                    "reward": rewards[local_i],
                    "hidden_state": hidden_states[local_i : local_i + 1],
                    "value": values[local_i],
                    "policy": policies[local_i : local_i + 1],
                    "to_play": to_plays[local_i : local_i + 1],
                    "rh": rh_news[local_i : local_i + 1],
                    "rc": rc_news[local_i : local_i + 1],
                }

        if afterstate_inputs:
            states = torch.cat([x["state"] for x in afterstate_inputs], dim=0)
            actions = (
                torch.tensor([x["action"] for x in afterstate_inputs])
                .to(self.device)
                .float()
                .unsqueeze(1)
            )

            afterstates, values, code_probs_batch = inference_fns["afterstate"](
                states, actions, model=inference_model
            )

            for local_i, x in enumerate(afterstate_inputs):
                idx = x["idx"]
                sim_data[idx]["result"] = {
                    "afterstate": afterstates[local_i : local_i + 1],
                    "value": values[local_i],
                    "code_probs": code_probs_batch[local_i : local_i + 1],
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
            # path has [Root, Child1, Child2, ...]
            # action_path has [Act1, Act2, ...]
            # We iterate up to len(action_path)
            for i in range(len(d["action_path"])):
                parent = path[i]
                action = d["action_path"][i]
                if isinstance(action, torch.Tensor):
                    act_idx = action.item()
                else:
                    act_idx = action
                parent.child_visits[act_idx] -= 1
                # No need to invalidate v_mix again, handled above (parent in path)

        # B. Backpropagation Phase
        for d in sim_data:
            path = d["path"]
            node = d["node"]

            res = d.get("result")
            if not res:
                continue

            to_play_for_backprop = None

            if isinstance(node, DecisionNode):
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
                    reward = reward.item()
                    value = value.item()

                to_play = int(res["to_play"].argmax().item())
                to_play_for_backprop = to_play

                is_reset = d["horizon_index"] == 0
                rh = res["rh"]
                rc = res["rc"]

                if self.config.value_prefix and is_reset:
                    rh = torch.zeros_like(rh).to(self.device)
                    rc = torch.zeros_like(rc).to(self.device)

                policy = res["policy"][0]
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
                    hidden_state=res["hidden_state"],
                    reward=reward,
                    value=value,
                    reward_h_state=rh,
                    reward_c_state=rc,
                    is_reset=is_reset,
                )

            elif isinstance(node, ChanceNode):
                value = res["value"]
                if self.config.support_range:
                    if value.numel() > 1:
                        value = support_to_scalar(
                            value, self.config.support_range
                        ).item()
                    else:
                        value = value.item()
                else:
                    value = value.item()

                to_play_for_backprop = d["parent"].to_play

                node.expand(
                    to_play=d["parent"].to_play,
                    afterstate=res["afterstate"],
                    network_value=value,
                    code_probs=res["code_probs"][0],
                    reward_h_state=d["parent"].reward_h_state,
                    reward_c_state=d["parent"].reward_c_state,
                )

            self.backpropagator.backpropagate(
                path,
                d["action_path"],
                value,
                to_play_for_backprop,
                min_max_stats,
                self.config,
            )
