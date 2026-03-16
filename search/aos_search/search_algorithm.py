import torch
from search.aos_search.search_factories import build_search_pipeline


class ModularSearch:
    def __init__(self, config, device, num_actions):
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self._run_mcts = build_search_pipeline(config, device, num_actions)

        compile_cfg = config.compilation
        if compile_cfg.enabled:
            # Compile the raw mathematical search procedure.
            # We don't directly compile the wrapper 'run' or 'run_vectorized' loops because
            # they handle dictionary unbatching, python iteration, and CPU synchronization
            # that cause dynamic graph breaks.
            self._run_mcts = torch.compile(
                self._run_mcts, fullgraph=compile_cfg.fullgraph
            )

    def run(
        self,
        observation,
        info,
        agent_network,
        trajectory_action=None,
        exploration=True,
    ):
        num_sims = self.config.num_simulations
        batch_size = self.config.search_batch_size
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
        num_sims = self.config.num_simulations
        batch_size = self.config.search_batch_size
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
