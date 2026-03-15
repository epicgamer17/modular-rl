import torch
from typing import Dict, List, Union, Optional
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

    def _vectorize_info(
        self, info: Union[Dict, List[Dict]], B: int
    ) -> Dict[str, torch.Tensor]:
        """Convert any info format into a dict of [B, ...] tensors for MCTSPipeline."""
        # 1. Standardize to a list of dicts
        if isinstance(info, dict):
            # If it's a dict containing lists (like pufferlib info), we might need to handle it.
            # But the current codebase often passes a single info dict or a list of dicts.
            if "player" in info and not isinstance(
                info["player"], (list, torch.Tensor)
            ):
                info_list = [info] * B
            elif "infos_list" in info:
                info_list = info["infos_list"]
            else:
                # Handle dict-of-lists/tensors
                info_list = None
        else:
            info_list = info

        # 2. Extract player_id
        if info_list is not None:
            assert all(
                "player" in i for i in info_list
            ), "Every info dict must contain 'player'. Got keys: " + str(
                list(info_list[0].keys())
            )
            player_ids = torch.tensor(
                [i["player"] for i in info_list],
                dtype=torch.int8,
                device=self.device,
            )

            # 3. Extract legal_moves mask
            legal_moves_list = [i.get("legal_moves", None) for i in info_list]
            if any(m is not None for m in legal_moves_list):
                mask = torch.zeros(
                    (B, self.num_actions), dtype=torch.bool, device=self.device
                )
                for b, moves in enumerate(legal_moves_list):
                    if moves is not None:
                        if torch.is_tensor(moves):
                            if moves.dim() == 1 and moves.shape[0] == self.num_actions:
                                mask[b] = moves.to(torch.bool)
                            else:
                                # Fallback for index tensor or other shapes
                                mask[b, moves] = True
                        else:
                            for a in moves:
                                mask[b, a] = True
                return {"player": player_ids, "legal_moves": mask}
            return {"player": player_ids}
        else:
            # Handle already-batched dict
            assert "player" in info, "info must contain 'player'. Got keys: " + str(
                list(info.keys())
            )
            p_val = info["player"]
            if not torch.is_tensor(p_val):
                p_val = torch.as_tensor(p_val, dtype=torch.int8, device=self.device)

            out = {"player": p_val}
            if "legal_moves" in info:
                lm = info["legal_moves"]
                if not torch.is_tensor(lm):
                    # Fallback conversion
                    mask = torch.zeros(
                        (B, self.num_actions), dtype=torch.bool, device=self.device
                    )
                    if (
                        isinstance(lm, list)
                        and len(lm) > 0
                        and not isinstance(lm[0], list)
                    ):
                        # Single list of indices for batch size 1
                        mask[0, lm] = True
                    else:
                        for b, moves in enumerate(lm):
                            for a in moves:
                                mask[b, a] = True
                    out["legal_moves"] = mask
                else:
                    # If it's [A], unsqueeze to [B, A]
                    if lm.dim() == 1 and B > 1:
                        lm = lm.unsqueeze(0).expand(B, -1)
                    elif lm.dim() == 1 and B == 1:
                        lm = lm.unsqueeze(0)
                    out["legal_moves"] = lm.to(dtype=torch.bool, device=self.device)
            return out

    def run(
        self,
        observation,
        info,
        agent_network,
        trajectory_action=None,
        exploration=True,
    ):
        batched_obs = (
            observation.to(self.device)
            if torch.is_tensor(observation)
            else torch.as_tensor(observation, device=self.device)
        )
        B = batched_obs.shape[0]

        # Pre-vectorize info to avoid graph breaks
        batched_info_t = self._vectorize_info(info, B)

        so = self._run_mcts(
            batched_obs,
            batched_info_t,
            agent_network,
            trajectory_actions=(
                trajectory_action if trajectory_action is not None else None
            ),
        )

        return (
            so.root_values[0].item(),
            so.exploratory_policy[0].cpu(),
            so.target_policy[0].cpu(),
            so.best_actions[0].item(),
            {
                "network_value": so.root_values[0].item(),
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
        B = (
            batched_obs.shape[0]
            if isinstance(batched_obs, torch.Tensor)
            else len(batched_obs)
        )

        # Pre-vectorize info to avoid graph breaks
        batched_info_t = self._vectorize_info(batched_info, B)

        so = self._run_mcts(
            batched_obs,
            batched_info_t,
            agent_network,
            trajectory_actions=trajectory_actions,
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
