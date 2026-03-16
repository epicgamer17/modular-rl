"""God-mode simulator for offline imitation learning from colonist.io replays."""

import sys
import os
import math
from typing import Optional
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0, "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron"
)

from custom_gym_envs.envs.catan import env as catan_env_factory
from catanatron.models.map import init_node_production, init_port_nodes_cache
from catanatron.models.actions import generate_playable_actions


def _dice_pair(forced_roll: int) -> tuple[int, int]:
    d1 = max(1, min(forced_roll - 1, 6))
    d2 = forced_roll - d1
    assert 1 <= d2 <= 6, f"Cannot represent roll {forced_roll} as valid dice pair"
    return (d1, d2)


class GodModeStepper:
    def __init__(self, **env_kwargs) -> None:
        self.env = catan_env_factory(**env_kwargs)
        self.env.reset()
        self._pending_resource_corrections = {}
        self._pending_dev_corrections = {}

    def sync_settings(self, settings: dict):
        if not settings:
            return
        game = self.env.unwrapped.game
        if "cardDiscardLimit" in settings:
            game.state.discard_limit = int(settings["cardDiscardLimit"])
        if "victoryPointsToWin" in settings:
            vps = int(settings["victoryPointsToWin"])
            game.vps_to_win = vps
            self.env.unwrapped.vps_to_win = vps

    def set_board_layout(self, board_config: dict, initial_state: dict = None):
        from offline_data.coordinate_mappings import WEB_TILE_TO_CATAN_COORD

        WEB_RES_TO_CATAN = {
            1: "WOOD",
            2: "BRICK",
            3: "SHEEP",
            4: "WHEAT",
            5: "ORE",
            0: None,
        }
        game = self.env.unwrapped.game
        desert_coord = None

        for web_id, tile_info in board_config.items():
            web_id = int(web_id)
            coord = WEB_TILE_TO_CATAN_COORD.get(web_id)
            if coord is None:
                continue
            res_val = tile_info.get("type", 0)
            res = WEB_RES_TO_CATAN.get(res_val)
            num = tile_info.get("diceNumber", 0) or None
            tile = game.state.board.map.land_tiles[coord]
            tile.resource = res
            tile.number = num

        game.state.board.map.node_production = init_node_production(
            game.state.board.map.adjacent_tiles
        )
        for coord, tile in game.state.board.map.land_tiles.items():
            if tile.resource is None:
                desert_coord = coord
                break
        if desert_coord is not None:
            game.state.board.robber_coordinate = desert_coord
        if hasattr(self.env.unwrapped, "viewer"):
            self.env.unwrapped.viewer = None

    def sync_ports(self, map_state: dict) -> None:
        PORT_TYPE_TO_RESOURCE: dict[int, str | None] = {
            1: None,
            2: "WOOD",
            3: "BRICK",
            4: "SHEEP",
            5: "WHEAT",
            6: "ORE",
        }
        port_states: dict = map_state.get("portEdgeStates", {})
        board = self.env.unwrapped.game.state.board

        def get_angle(q, r):
            cx = 1.73205 * (q + r / 2.0)
            cy = 1.5 * r
            return math.atan2(cy, cx)

        colonist_ports = []
        for _pid, pe in port_states.items():
            q, r = pe["x"], pe["y"]
            angle = get_angle(q, r)
            resource = PORT_TYPE_TO_RESOURCE.get(pe.get("type", 1), None)
            colonist_ports.append((angle, resource))

        colonist_ports.sort(key=lambda x: x[0])
        catan_ports = []
        for coord, tile in board.map.tiles.items():
            if type(tile).__name__ == "Port":
                x, y, z = coord
                angle = get_angle(x, z)
                catan_ports.append((angle, tile))

        catan_ports.sort(key=lambda x: x[0])
        for (_, res), (_, tile) in zip(colonist_ports, catan_ports):
            tile.resource = res

        board.map.port_nodes = init_port_nodes_cache(board.map.tiles)
        board.player_port_resources_cache = {}

    def sync_resources(self, player_states_diff: dict, play_order: list):
        for p_key, state in player_states_diff.items():
            p_color = int(p_key)
            if p_color not in play_order:
                continue
            p_idx = play_order.index(p_color)

            if "resourceCards" in state and "cards" in state["resourceCards"]:
                cards = state["resourceCards"]["cards"]
                if cards is not None:
                    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    for res_enum in cards:
                        if 1 <= res_enum <= 5:
                            counts[res_enum] += 1
                    self._pending_resource_corrections[p_idx] = counts

            if "developmentCards" in state:
                dev_data = state["developmentCards"]
                if isinstance(dev_data, dict):
                    counts = {}
                    if "cards" in dev_data and isinstance(dev_data["cards"], list):
                        for dev_enum in dev_data["cards"]:
                            counts[dev_enum] = counts.get(dev_enum, 0) + 1
                    else:
                        for web_dev_idx, count in dev_data.items():
                            if str(web_dev_idx) != "cards":
                                counts[int(web_dev_idx)] = int(count)
                    if counts:
                        self._pending_dev_corrections[p_idx] = counts

    def _update_true_bank(self):
        """Calculates the exact bank state based on the conservation of 19 cards per resource."""
        game = self.env.unwrapped.game
        true_bank = [19, 19, 19, 19, 19]
        num_players = len(game.state.colors)
        res_names_in_order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

        for i, res_name in enumerate(res_names_in_order):
            total_in_hands = sum(
                game.state.player_state.get(f"P{p}_{res_name}_IN_HAND", 0)
                for p in range(num_players)
            )
            true_bank[i] = max(0, 19 - total_in_hands)

        game.state.resource_freqdeck = true_bank

    def step_and_override(
        self,
        action_idx: int,
        forced_roll: Optional[int],
        forced_dev_card: Optional[str],
    ) -> np.ndarray:
        game = self.env.unwrapped.game

        if forced_dev_card is not None:
            deck = game.state.development_listdeck
            if forced_dev_card in deck:
                deck.remove(forced_dev_card)
                deck.append(forced_dev_card)

        # Ensure the bank is strictly accurate before generating valid actions
        self._update_true_bank()
        game.playable_actions = generate_playable_actions(game.state)

        valid_actions = self.env.unwrapped._get_valid_action_indices()

        if action_idx not in valid_actions:
            from custom_gym_envs.envs.catan import ACTIONS_ARRAY

            action_type, value = ACTIONS_ARRAY[action_idx]
            print(f"\n{'='*70}")
            print(f"🚨 DESYNC DETECTED: ACTION REJECTED BY CATANATRON 🚨")
            print(f"{'='*70}")
            print(
                f"Attempted Action from Colonist : {action_type.name} {value} (Index: {action_idx})"
            )
            print(f"PettingZoo Expected Agent      : {self.env.agent_selection}")
            print(f"Catanatron Internal Turn Color : {game.state.current_color()}")
            print(f"\n--- Catanatron Game Phase ---")
            stages = getattr(game.state, "turn_stages", [])
            print(
                f"Turn Stages Stack   : {[stage.name if hasattr(stage, 'name') else str(stage) for stage in stages]}"
            )
            print(
                f"Game Num Turns      : {getattr(game.state, 'num_turns', 'Unknown')}"
            )
            print(f"Discard Limit       : {game.state.discard_limit}")
            print(f"Bank                : {game.state.resource_freqdeck}")
            print(f"\n--- Valid Actions Allowed By Catanatron Right Now ---")
            for va in valid_actions:
                v_type, v_val = ACTIONS_ARRAY[va]
                print(f"  - [{va}] {v_type.name} {v_val}")
            print(f"{'='*70}\n")
            raise ValueError(
                "Stopping execution to review the diagnostic report above."
            )

        if forced_roll is not None:
            dice = _dice_pair(forced_roll)
            mock_roll = lambda: dice
            with patch("catanatron.apply_action.roll_dice", mock_roll):
                self.env.step(action_idx)
        else:
            self.env.step(action_idx)

        obs_dict, _rew, _term, _trunc, _info = self.env.last()
        return obs_dict["observation"]

    def flush_corrections(self):
        """Applies all pending JSON state corrections.
        MUST be called at the very end of an event, AFTER all actions for that event
        have been stepped through, so we don't overwrite intermediate hands."""
        game = self.env.unwrapped.game
        RES_KEYS = {1: "WOOD", 2: "BRICK", 3: "SHEEP", 4: "WHEAT", 5: "ORE"}
        DEV_KEYS = {
            11: "KNIGHT",
            12: "MONOPOLY",
            14: "ROAD_BUILDING",
            15: "YEAR_OF_PLENTY",
            13: "VICTORY_POINT",
        }
        needs_update = False

        if self._pending_resource_corrections:
            for p_idx, counts in self._pending_resource_corrections.items():
                for web_res_idx, count in counts.items():
                    res_name = RES_KEYS[web_res_idx]
                    catan_key = f"P{p_idx}_{res_name}_IN_HAND"
                    game.state.player_state[catan_key] = count
            self._pending_resource_corrections.clear()
            needs_update = True

        if self._pending_dev_corrections:
            for p_idx, counts in self._pending_dev_corrections.items():
                for web_dev_idx, count in counts.items():
                    dev_name = DEV_KEYS.get(web_dev_idx)
                    if dev_name:
                        catan_key = f"P{p_idx}_{dev_name}_IN_HAND"
                        game.state.player_state[catan_key] = count
            self._pending_dev_corrections.clear()
            needs_update = True

        if needs_update:
            self._update_true_bank()
            game.playable_actions = generate_playable_actions(game.state)


if __name__ == "__main__":
    pass
