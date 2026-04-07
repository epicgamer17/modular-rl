import torch
from typing import List, Optional
from search.backends.aos_search.search_factories import build_search_pipeline


class ModularSearch:
    def __init__(
        self,
        device: torch.device,
        num_actions: int,
        num_simulations: int = 50,
        max_search_depth: int = 5,
        max_nodes: int = 512,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652.0,
        discount_factor: float = 1.0,
        use_dirichlet: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
        backprop_method: str = "average",
        policy_extraction: str = "visit_count",
        scoring_method: str = "ucb",
        search_batch_size: int = 1,
        num_codes: int = 0,
        gumbel_cvisit: float = 50.0,
        gumbel_cscale: float = 1.0,
        use_virtual_mean: bool = False,
        virtual_loss: float = 0.0,
        bootstrap_method: str = "parent_value",
        num_players: int = 1,
        use_sequential_halving: bool = False,
        gumbel_m: int = 8,
        known_bounds: Optional[List[float]] = None,
        min_max_epsilon: float = 1e-8,
        internal_decision_modifier: str = "none",
        internal_chance_modifier: str = "none",
        compile_enabled: bool = False,
        compile_fullgraph: bool = False,
    ):
        self.device = device
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        self.search_batch_size = search_batch_size

        self._run_mcts = build_search_pipeline(
            device=device,
            num_actions=num_actions,
            num_simulations=num_simulations,
            max_search_depth=max_search_depth,
            max_nodes=max_nodes,
            pb_c_init=pb_c_init,
            pb_c_base=pb_c_base,
            discount_factor=discount_factor,
            use_dirichlet=use_dirichlet,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
            backprop_method=backprop_method,
            policy_extraction=policy_extraction,
            scoring_method=scoring_method,
            search_batch_size=search_batch_size,
            num_codes=num_codes,
            gumbel_cvisit=gumbel_cvisit,
            gumbel_cscale=gumbel_cscale,
            use_virtual_mean=use_virtual_mean,
            virtual_loss=virtual_loss,
            bootstrap_method=bootstrap_method,
            num_players=num_players,
            use_sequential_halving=use_sequential_halving,
            gumbel_m=gumbel_m,
            known_bounds=known_bounds,
            min_max_epsilon=min_max_epsilon,
            internal_decision_modifier=internal_decision_modifier,
            internal_chance_modifier=internal_chance_modifier,
        )

        if compile_enabled:
            # Compile the raw mathematical search procedure.
            self._run_mcts = torch.compile(self._run_mcts, fullgraph=compile_fullgraph)

    def run(
        self,
        observation,
        info,
        agent_network,
        trajectory_action=None,
        exploration=True,
    ):
        num_sims = self.num_simulations
        batch_size = max(1, self.search_batch_size)
        effective_depth = num_sims / batch_size
        if effective_depth < 5 and num_sims > batch_size:
            import warnings

            warnings.warn(
                f"AOS Search depth is very shallow ({effective_depth:.1f}). "
                f"Consider decreasing search_batch_size ({batch_size}) "
                f"or increasing num_simulations ({num_sims}).",
                RuntimeWarning,
                stacklevel=2,
            )

        batched_obs = observation
        batched_info = info

        B = batched_obs.shape[0]
        assert (
            B == 1
        ), f"AOS modular_search.run() expects exactly 1 observation, got {B}."

        # Normalize player to a [1] tensor so both int and tensor inputs are accepted.
        player_raw = info["player"]
        if not torch.is_tensor(player_raw):
            player_raw = torch.tensor([player_raw], dtype=torch.int8)
        assert (
            player_raw.shape[0] == 1
        ), f"AOS modular_search.run() player batch mismatch: {player_raw.shape}"
        batched_info = {**info, "player": player_raw}

        # Extract scalar player for logging / metadata
        to_play = int(player_raw[0].item())
        batched_to_play = to_play
        so = self._run_mcts(batched_obs, batched_info, agent_network)

        return (
            so.root_values[0].item(),
            so.exploratory_policy[0].cpu(),
            so.target_policy[0].cpu(),
            so.best_actions[0].item(),
            {
                "network_value": so.root_values[
                    0
                ].item(),  # AOS returns the network value in tree for now
                "search_value": so.root_values[0].item(),
                "search_policy": so.target_policy[0].cpu().tolist(),
            },
        )

    def run_vectorized(
        self,
        batched_obs,
        batched_info,
        agent_network,
        trajectory_actions=None,
        exploration=True,
    ):
        num_sims = self.num_simulations
        batch_size = max(1, self.search_batch_size)
        effective_depth = num_sims / batch_size
        if effective_depth < 5 and num_sims > batch_size:
            import warnings

            warnings.warn(
                f"AOS Search depth is very shallow ({effective_depth:.1f}). "
                f"Consider decreasing search_batch_size ({batch_size}) "
                f"or increasing num_simulations ({num_sims}).",
                RuntimeWarning,
                stacklevel=2,
            )

        B = batched_obs.shape[0]
        # Normalize list-of-dicts to dict-of-tensors (mirrors Python backend behavior)
        if isinstance(batched_info, list):
            batched_info = {
                "player": torch.tensor(
                    [info["player"] for info in batched_info], dtype=torch.int8
                ),
                **{
                    k: [info[k] for info in batched_info]
                    for k in batched_info[0]
                    if k != "player"
                },
            }
        assert (
            batched_info["player"].shape[0] == B
        ), f"AOS modular_search.run_vectorized() batch mismatch: {B} vs {batched_info['player'].shape[0]}"

        so = self._run_mcts(
            batched_obs,
            batched_info,
            agent_network,
            trajectory_actions=trajectory_actions,
        )

        B = batched_obs.shape[0]
        root_values = so.root_values.cpu().tolist()
        exploratory_policies = [so.exploratory_policy[i].cpu() for i in range(B)]
        target_policies = [so.target_policy[i].cpu() for i in range(B)]
        best_actions = so.best_actions.cpu().tolist()
        search_metadata_list = [
            {
                "network_value": root_values[i],
                "search_value": root_values[i],
                "search_policy": target_policies[i].tolist(),
            }
            for i in range(B)
        ]

        return (
            root_values,
            exploratory_policies,
            target_policies,
            best_actions,
            search_metadata_list,
        )
