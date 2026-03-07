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
        to_play,
        agent_network,
        trajectory_action=None,
        exploration=True,
    ):
        batched_obs = (
            observation.unsqueeze(0)
            if torch.is_tensor(observation)
            else torch.tensor([observation], device=self.device)
        )
        batched_info = {k: [v] for k, v in info.items()} if info else {}
        if info and "legal_moves" in info:
            batched_info["legal_moves"] = [info["legal_moves"]]

        batched_to_play = torch.tensor([to_play], dtype=torch.int8, device=self.device)
        so = self._run_mcts(batched_obs, batched_info, batched_to_play, agent_network)

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
        batched_to_play,
        agent_network,
        trajectory_actions=None,
        exploration=True,
    ):
        batched_to_play_t = (
            batched_to_play
            if torch.is_tensor(batched_to_play)
            else torch.tensor(batched_to_play, dtype=torch.int8, device=self.device)
        )
        so = self._run_mcts(
            batched_obs,
            batched_info,
            batched_to_play_t.to(self.device),
            agent_network,
            trajectory_actions=trajectory_actions,
        )

        B = (
            batched_obs.shape[0]
            if isinstance(batched_obs, torch.Tensor)
            else len(batched_obs)
        )
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
