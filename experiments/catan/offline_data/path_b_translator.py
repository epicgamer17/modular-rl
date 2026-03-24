"""Path B: direct JSON → observation tensor translation, bypassing CatanEnv entirely.

JsonStateTracker incrementally applies colonist.io stateChange diffs to an
in-memory board representation. to_tensor() produces a numpy array with the
same shape and channel layout as catanatron's create_board_tensor() called with:

    spatial_encoding="axial",
    include_validity_mask=True,
    include_last_roll=True,
    include_game_phase=True,
    include_bank_state=True,
    include_road_distance=True,
    channels_first=True

Shape: (57, 22, 14) for 2 players.

Channel layout (2 players)
───────────────────────────
  0- 5  player building planes (settlement, city, road × n_players, ME-first)
  6-11  tile resource planes (WOOD, BRICK, SHEEP, WHEAT, ORE, DESERT)
 12-21  tile dice-number planes (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
    22  robber
 23-28  port planes (WOOD, BRICK, SHEEP, WHEAT, ORE, 3:1)
    29  validity mask
 30-40  last-roll one-hot (rolls 2-12)
 41-44  game-phase (IS_DISCARDING, IS_MOVING_ROBBER, P0_HAS_ROLLED, P1_HAS_ROLLED)
 45-54  bank-state (5 normalised counts + 5 empty indicators, RESOURCES order)
 55-56  road-distance per player (P0, P1)

Colonist.io ↔ catanatron mappings are in offline_data/coordinate_mappings.py.
"""

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0,
    "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron",
)

from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
from catanatron.models.enums import RESOURCES
from catanatron.models.decks import RESOURCE_FREQDECK_INDEXES
from catanatron.gym.board_tensor_features import (
    get_axial_node_edge_maps,
    get_axial_validity_mask,
    AXIAL_WIDTH,
    AXIAL_HEIGHT,
)

from coordinate_mappings import (
    WEB_CORNER_TO_CATAN_NODE,
    WEB_EDGE_TO_CATAN_EDGE,
    WEB_TILE_TO_CATAN_COORD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CATAN_MAP = CatanMap.from_template(BASE_MAP_TEMPLATE)
_NODE_MAP, _EDGE_MAP = get_axial_node_edge_maps()
_VALIDITY_MASK = get_axial_validity_mask(_CATAN_MAP)

# Colonist.io resource enum → catanatron resource string (0 = desert)
_COLONIST_RES: dict[int, Optional[str]] = {
    0: None,
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}

# catanatron resource string → channel offset within the 6-resource block
_RES_CH: dict[Optional[str], int] = {r: i for i, r in enumerate(RESOURCES)}
_RES_CH[None] = 5  # desert → channel offset 5

MAX_BANK = 19
MAX_ROAD = 15
W = AXIAL_WIDTH * 2  # 22
H = AXIAL_HEIGHT * 2  # 14


def _iter_players(play_order: list[int], p0_color: int):
    """Yield (relative_index, colonist_color) starting from p0_color."""
    start = play_order.index(p0_color)
    n = len(play_order)
    for i in range(n):
        yield i, play_order[(start + i) % n]


def _tile_center(coord: tuple) -> Optional[tuple[int, int]]:
    """Return (x, y) integer pixel centre of a tile in the axial grid, or None."""
    tile = _CATAN_MAP.land_tiles.get(coord)
    if tile is None:
        return None
    xs = [_NODE_MAP[n][0] for n in tile.nodes.values() if n in _NODE_MAP]
    ys = [_NODE_MAP[n][1] for n in tile.nodes.values() if n in _NODE_MAP]
    if not xs:
        return None
    return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))


# Pre-compute tile centres for all 19 land tiles.
_TILE_CENTER: dict[tuple, tuple[int, int]] = {}
for _coord in _CATAN_MAP.land_tiles:
    _c = _tile_center(_coord)
    if _c is not None:
        _TILE_CENTER[_coord] = _c

# Pre-compute port node positions
_PORT_NODES: dict[Optional[str], list[tuple[int, int]]] = {}
for _res, _nids in _CATAN_MAP.port_nodes.items():
    _PORT_NODES[_res] = [_NODE_MAP[n] for n in _nids if n in _NODE_MAP]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class JsonStateTracker:
    """Incrementally tracks colonist.io game state from JSON stateChange diffs.

    Args:
        play_order:   Colonist.io player color integers in turn order (from
                      data['playOrder']). Example: [5, 1].
        board_config: Dict mapping web_tile_id (int) → (colonist_resource_type,
                      dice_number). Obtained by pre-scanning all tileInfo events
                      in the replay. Missing tiles contribute zeros to the tensor.
    """

    def __init__(
        self,
        play_order: list[int],
        board_config: Optional[dict[int, tuple[int, int]]] = None,
        discard_limit: int = 7,
    ) -> None:
        self.play_order = play_order
        self.n = len(play_order)
        self.discard_limit = discard_limit

        # ----- Static board from board_config --------------------------------
        # tile_data[catan_coord] = (resource_str_or_None, dice_number_or_None)
        self.tile_data: dict[tuple, tuple[Optional[str], Optional[int]]] = {}
        self._desert_coord: Optional[tuple] = None

        if board_config:
            for web_tile_id, (res_enum, dice_num) in board_config.items():
                coord = WEB_TILE_TO_CATAN_COORD[web_tile_id]
                res_str = _COLONIST_RES.get(res_enum)
                num = None if dice_num == 0 else dice_num
                self.tile_data[coord] = (res_str, num)
                if res_enum == 0:
                    self._desert_coord = coord

        # ----- Dynamic state -------------------------------------------------
        # node_buildings[catan_node] = (colonist_color, building_type)
        # building_type: 1=settlement, 2=city
        self.node_buildings: dict[int, tuple[int, int]] = {}

        # edge_buildings[catan_edge] = colonist_color
        # catan_edge is a sorted (a, b) tuple matching WEB_EDGE_TO_CATAN_EDGE values
        self.edge_buildings: dict[tuple[int, int], int] = {}

        # Robber starts on the desert
        self.robber_coord: Optional[tuple] = self._desert_coord

        # Bank: colonist resource enum (1-5) → absolute count in bank
        self.bank: dict[int, int] = {r: MAX_BANK for r in range(1, 6)}

        # Player resources: colonist_color → colonist_res_enum → count
        self.player_res: dict[int, dict[int, int]] = {
            c: {r: 0 for r in range(1, 6)} for c in play_order
        }

        # --- ADD THIS: Track absolute card counts to handle hidden opponents ---
        self.player_card_counts: dict[int, int] = {c: 0 for c in play_order}

        # --- NEW FIX: Explicitly track who still needs to discard ---
        self.players_needing_discard: set[int] = set()
        # ---------------------------------------------------------------------

        # Road lengths (from mechanicLongestRoadState).
        # Zeroed during initial placement to match catanatron's behaviour
        # (catanatron does not call maintain_longest_road during initial build phase).
        self.road_lengths: dict[int, int] = {c: 0 for c in play_order}

        # Initial build phase: True until all n players have placed 2 settlements.
        # Road lengths read from mechanicLongestRoadState are ignored while True.
        self._initial_phase: bool = True
        self._total_settlements_placed: int = 0

        # Game phase
        self.is_discarding: bool = False
        self.is_moving_robber: bool = False
        self.has_rolled: dict[int, bool] = {c: False for c in play_order}

        # Last dice roll (sum 2-12, or None if not yet rolled this turn)
        self.last_roll: Optional[int] = None

    # ---------------------------------------------------------------------- #
    # update                                                                  #
    # ---------------------------------------------------------------------- #

    def update(self, state_change_diff: dict) -> None:
        """Apply one colonist.io stateChange diff to the tracker's internal state.

        Reads mapState, playerStates, bankState, mechanicRobberState,
        mechanicLongestRoadState, diceState, currentState and gameLogState
        to keep all tracked state current.
        """
        sc = state_change_diff

        # ---- Map state: buildings and roads ---------------------------------
        ms = sc.get("mapState", {})
        prev_settlement_count = sum(
            1 for _, btype in self.node_buildings.values() if btype == 1
        )
        for cid_str, cdata in ms.get("tileCornerStates", {}).items():
            web_corner = int(cid_str)
            if web_corner not in WEB_CORNER_TO_CATAN_NODE:
                continue
            catan_node = WEB_CORNER_TO_CATAN_NODE[web_corner]
            # FIX: Retrieve existing state so partial diffs don't erase it
            existing_owner, existing_btype = self.node_buildings.get(
                catan_node, (None, 1)
            )

            owner = cdata.get("owner", existing_owner)
            btype = cdata.get(
                "buildingType", existing_btype
            )  # 1=settlement, 2=city

            if owner is not None:
                self.node_buildings[catan_node] = (owner, btype)

        # Track total settlements placed to detect end of initial build phase.
        # catanatron ends initial phase when all n players have placed 2 settlements
        # (num_buildings == 2 * n_players).  Road lengths are zeroed until then.
        if self._initial_phase:
            new_settlement_count = sum(
                1 for _, btype in self.node_buildings.values() if btype == 1
            )
            self._total_settlements_placed += (
                new_settlement_count - prev_settlement_count
            )
            if self._total_settlements_placed >= 2 * self.n:
                self._initial_phase = False

        for eid_str, edata in ms.get("tileEdgeStates", {}).items():
            web_edge = int(eid_str)
            if web_edge not in WEB_EDGE_TO_CATAN_EDGE:
                continue
            catan_edge = WEB_EDGE_TO_CATAN_EDGE[web_edge]
            # FIX: Fallback to existing owner for edge updates
            owner = edata.get("owner", self.edge_buildings.get(catan_edge))
            if owner is not None:
                self.edge_buildings[catan_edge] = owner

        # ---- Robber location ------------------------------------------------
        mrs = sc.get("mechanicRobberState", {})
        if "locationTileIndex" in mrs:
            web_tile = mrs["locationTileIndex"]
            self.robber_coord = WEB_TILE_TO_CATAN_COORD.get(web_tile)

        # ---- Bank state (absolute delta dict, key=colonist res enum str) ----
        bs = sc.get("bankState", {})
        for res_str, count in bs.get("resourceCards", {}).items():
            res_enum = int(res_str)
            if res_enum in self.bank:
                self.bank[res_enum] = int(count)

        # ---- Player states --------------------------------------------------
        for pcolor_str, pdata in sc.get("playerStates", {}).items():
            pcolor = int(pcolor_str)
            if pcolor not in self.player_res:
                continue

            rc = pdata.get("resourceCards", {})

            # 1. Update absolute count (crucial for hidden opponents)
            if "count" in rc:
                self.player_card_counts[pcolor] = rc["count"]
            elif "length" in rc:
                self.player_card_counts[pcolor] = rc["length"]
            elif "total" in rc:
                self.player_card_counts[pcolor] = rc["total"]

            # 2. Update specific cards if they are public
            cards = rc.get("cards")
            if cards is not None:
                new_res = {r: 0 for r in range(1, 6)}
                for res_enum in cards:
                    if 1 <= res_enum <= 5:
                        new_res[res_enum] += 1
                self.player_res[pcolor] = new_res
                # Fallback just in case count wasn't provided
                if (
                    "count" not in rc
                    and "length" not in rc
                    and "total" not in rc
                ):
                    self.player_card_counts[pcolor] = sum(new_res.values())

        # ---- Road lengths ---------------------------------------------------
        # Mirror catanatron: road lengths stay at 0 during initial build phase.
        if not self._initial_phase:
            mlrs = sc.get("mechanicLongestRoadState", {})
            for pcolor_str, ldata in mlrs.items():
                pcolor = int(pcolor_str)
                if pcolor in self.road_lengths:
                    # FIX: Only update if explicitly in the diff to avoid zeroing it out
                    if "longestRoad" in ldata:
                        self.road_lengths[pcolor] = ldata["longestRoad"]

        # ---- Tile info (learn board layout from robber visits) --------------
        gls = sc.get("gameLogState", {})
        for k, v in gls.items():
            t = v.get("text", {})
            if isinstance(t, dict) and "tileInfo" in t and "locationTileIndex" in mrs:
                web_tile = mrs["locationTileIndex"]
                ti = t["tileInfo"]
                coord = WEB_TILE_TO_CATAN_COORD.get(web_tile)
                if coord is not None and coord not in self.tile_data:
                    res_enum = ti.get("resourceType", ti.get("tileType", 0))
                    dice_num = ti.get("diceNumber", 0)
                    res_str = _COLONIST_RES.get(res_enum)
                    num = None if dice_num == 0 else dice_num
                    self.tile_data[coord] = (res_str, num)
                    if res_enum == 0 and self._desert_coord is None:
                        self._desert_coord = coord
                        if self.robber_coord is None:
                            self.robber_coord = coord

        # ---- Dice / game phase ----------------------------------------------
        ds = sc.get("diceState", {})
        if ds.get("diceThrown"):
            # Reconstruct roll from gameLogState text (diceState may omit one die).
            roll = None
            for _, v in gls.items():
                t = v.get("text", {})
                if isinstance(t, dict) and t.get("type") == 10:
                    roll = t.get("firstDice", 0) + t.get("secondDice", 0)
                    break
            if roll is not None:
                self.last_roll = roll

                # FIX: Record exactly who needs to discard right when the 7 is rolled
                if roll == 7:
                    self.players_needing_discard = {
                        c
                        for c in self.play_order
                        if self.player_card_counts.get(c, 0) > self.discard_limit
                    }
                    if self.players_needing_discard:
                        self.is_discarding = True
                    else:
                        self.is_moving_robber = True

        for _, v in gls.items():
            t = v.get("text", {})
            if not isinstance(t, dict):
                continue
            ttype = t.get("type")
            pcolor = t.get("playerColor")

            # FIX: Playing a Knight instantly triggers the Robber phase
            if ttype == 20 and t.get("cardEnum") in (10, 11):
                self.is_moving_robber = True

            if ttype == 10 and pcolor in self.has_rolled:
                self.has_rolled[pcolor] = True
            elif ttype == 44:
                # END_TURN: reset per-turn flags for whoever just ended
                for c in self.play_order:
                    self.has_rolled[c] = False
                self.is_discarding = False
                self.is_moving_robber = False
                self.players_needing_discard.clear()  # Ensure list clears on turn end
            elif ttype == 60:
                self.is_discarding = True
            elif ttype == 55:
                # DISCARD completed; check if more players need to discard using reliable counts
                # FIX: Simply remove the player from the queue when they emit a discard event
                if pcolor in self.players_needing_discard:
                    self.players_needing_discard.remove(pcolor)

                if not self.players_needing_discard:
                    self.is_discarding = False
                    self.is_moving_robber = True
            elif ttype == 11:
                self.is_moving_robber = False
                self.is_discarding = False

    # ---------------------------------------------------------------------- #
    # to_tensor                                                               #
    # ---------------------------------------------------------------------- #

    def to_tensor(self, current_player_color: int) -> np.ndarray:
        """Build the board observation tensor from current tracked state.

        Returns channels-first numpy array of shape (C, W, H) matching the
        output of create_board_tensor(..., channels_first=True, spatial_encoding="axial",
        include_validity_mask=True, include_last_roll=True, include_game_phase=True,
        include_bank_state=True, include_road_distance=True).

        Args:
            current_player_color: Colonist.io color int of the acting player
                                  (determines ME/NEXT relative channel ordering).
        """
        n = self.n
        # 3n + 6 + 10 + 1 + 6 + 1 + 11 + (2+n) + 10 + n
        # = 29 + 1 + 11 + 2+n + 10 + n = 53 + 2n
        # For n=2: 57
        n_channels = 3 * n + 6 + 10 + 1 + 6 + 1 + 11 + (2 + n) + 10 + n
        planes = np.zeros((n_channels, W, H), dtype=np.float32)

        ordered = list(_iter_players(self.play_order, current_player_color))
        base = 3 * n

        # ------------------------------------------------------------------ #
        # 1. Player building planes (3*n channels)                           #
        # ------------------------------------------------------------------ #
        for rel_i, colonist_color in ordered:
            s_ch = 3 * rel_i  # settlement
            c_ch = 3 * rel_i + 1  # city
            r_ch = 3 * rel_i + 2  # road

            for node_id, (owner, btype) in self.node_buildings.items():
                if owner != colonist_color:
                    continue
                if node_id not in _NODE_MAP:
                    continue
                x, y = _NODE_MAP[node_id]
                if btype == 1:
                    planes[s_ch, x, y] = 1.0
                else:  # city
                    planes[c_ch, x, y] = 1.0

            for edge, owner in self.edge_buildings.items():
                if owner != colonist_color:
                    continue
                if edge not in _EDGE_MAP:
                    continue
                x, y = _EDGE_MAP[edge]
                planes[r_ch, x, y] = 1.0

        # ------------------------------------------------------------------ #
        # 2. Resource & dice channels (6 + 10) and robber (1)                #
        # ------------------------------------------------------------------ #
        for coord, (res_str, dice_num) in self.tile_data.items():
            center = _TILE_CENTER.get(coord)
            if center is None:
                continue
            cx, cy = center

            # Resource channel
            res_ch_offset = _RES_CH.get(res_str, 5)
            planes[base + res_ch_offset, cx, cy] = 1.0

            # Dice number channel
            if dice_num is not None and 2 <= dice_num <= 12 and dice_num != 7:
                dice_idx = dice_num - 2 if dice_num <= 6 else dice_num - 3
                planes[base + 6 + dice_idx, cx, cy] = 1.0

        # Robber channel
        robber_ch = base + 6 + 10
        if self.robber_coord is not None:
            center = _TILE_CENTER.get(self.robber_coord)
            if center is not None:
                planes[robber_ch, center[0], center[1]] = 1.0

        # ------------------------------------------------------------------ #
        # 3. Port channels (6)                                               #
        # ------------------------------------------------------------------ #
        port_base = base + 6 + 10 + 1
        for res_str, positions in _PORT_NODES.items():
            ch_offset = 5 if res_str is None else _RES_CH[res_str]
            ch = port_base + ch_offset
            for x, y in positions:
                planes[ch, x, y] = 1.0

        # ------------------------------------------------------------------ #
        # 4. Validity mask                                                    #
        # ------------------------------------------------------------------ #
        current_ch = port_base + 6  # == 29
        planes[current_ch] = _VALIDITY_MASK[:W, :H]
        current_ch += 1  # 30

        # ------------------------------------------------------------------ #
        # 5. Last roll one-hot (11 channels)                                 #
        # ------------------------------------------------------------------ #
        if self.last_roll is not None and 2 <= self.last_roll <= 12:
            planes[current_ch + (self.last_roll - 2), :, :] = 1.0
        current_ch += 11  # 41

        # ------------------------------------------------------------------ #
        # 6. Game phase (2 + n channels)                                     #
        # ------------------------------------------------------------------ #
        if self.is_discarding:
            planes[current_ch, :, :] = 1.0
        current_ch += 1  # 42

        if self.is_moving_robber:
            planes[current_ch, :, :] = 1.0
        current_ch += 1  # 43

        for rel_i, colonist_color in ordered:
            if self.has_rolled.get(colonist_color, False):
                planes[current_ch, :, :] = 1.0
            current_ch += 1
        # current_ch == 43 + n = 45 for n=2

        # ------------------------------------------------------------------ #
        # 7. Bank state (10 channels: 5 normalised + 5 empty flags)          #
        # ------------------------------------------------------------------ #
        for res_str in RESOURCES:
            res_enum = next(k for k, v in _COLONIST_RES.items() if v == res_str)
            count = self.bank.get(res_enum, 0)
            planes[current_ch, :, :] = count / MAX_BANK
            current_ch += 1

        for res_str in RESOURCES:
            res_enum = next(k for k, v in _COLONIST_RES.items() if v == res_str)
            count = self.bank.get(res_enum, 0)
            if count == 0:
                planes[current_ch, :, :] = 1.0
            current_ch += 1
        # current_ch == 45 + 10 = 55 for n=2

        # ------------------------------------------------------------------ #
        # 8. Road distance (n channels)                                      #
        # ------------------------------------------------------------------ #
        road_vals = [self.road_lengths.get(c, 0) for _, c in ordered]
        max_road = max(road_vals) if road_vals else 0
        for road_len in road_vals:
            dist = (max_road - road_len) / MAX_ROAD if max_road > 0 else 0.0
            planes[current_ch, :, :] = dist
            current_ch += 1

        return planes


# ---------------------------------------------------------------------------
# __main__: dry-run on ZxFhBPVr4KvuC3yN.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    REPLAY_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "experiments",
        "rainbowzero",
        "catan",
        "replays",
        "ZxFhBPVr4KvuC3yN.json",
    )

    with open(REPLAY_PATH) as f:
        data = json.load(f)["data"]

    events = data["eventHistory"]["events"]
    play_order: list[int] = data["playOrder"]

    # Pre-scan all events to collect as much board tile info as possible.
    board_config: dict[int, tuple[int, int]] = {}
    for ev in events:
        sc = ev.get("stateChange", {})
        mrs = sc.get("mechanicRobberState", {})
        if "locationTileIndex" not in mrs:
            continue
        web_tile = mrs["locationTileIndex"]
        gls = sc.get("gameLogState", {})
        for _, v in gls.items():
            t = v.get("text", {})
            if isinstance(t, dict) and "tileInfo" in t:
                ti = t["tileInfo"]
                res_enum = ti.get("resourceType", ti.get("tileType", 0))
                dice_num = ti.get("diceNumber", 0)
                board_config[web_tile] = (res_enum, dice_num)
                break

    print(f"Board tiles learned from pre-scan: {len(board_config)}/19")

    tracker = JsonStateTracker(play_order, board_config=board_config)

    # Determine acting player (mirrors json_parser logic).
    from offline_data.json_parser import _texts_sorted

    current_turn_color: int = play_order[0]
    step = 0
    TARGET = 30

    print(
        f"{'#':<4}  {'ev':<5}  {'current_player':<16}  " f"{'obs_shape':<16}  obs_sum"
    )
    print("-" * 75)

    for ev_idx, event in enumerate(events):
        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        # Determine acting player before updating current_turn_color.
        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        # Apply diff to tracker.
        tracker.update(sc)

        # Update current_turn_color from currentState (after extracting actor).
        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        # Build tensor for the acting player.
        obs = tracker.to_tensor(acting_player)
        print(
            f"{step:<4}  ev{ev_idx:<4}  p{acting_player:<15}  "
            f"{str(obs.shape):<16}  {obs.sum():.2f}"
        )
        step += 1
        if step >= TARGET:
            break
