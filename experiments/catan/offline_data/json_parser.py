"""Parse colonist.io game log events into catanatron action indices and RNG seeds."""

import sys
import os
from typing import Optional, List, Tuple
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0,
    "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron",
)

from catanatron.models.enums import ActionType as AT
from custom_gym_envs.envs.catan import ACTIONS_ARRAY
from coordinate_mappings import (
    WEB_CORNER_TO_CATAN_NODE,
    WEB_EDGE_TO_CATAN_EDGE,
    WEB_TILE_TO_CATAN_COORD,
)

_RESOURCE_MAP: dict[int, str] = {
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}

_DEVCARD_MAP: dict[int, str] = {
    11: "KNIGHT",
    12: "VICTORY_POINT",
    13: "MONOPOLY",
    14: "ROAD_BUILDING",
    15: "YEAR_OF_PLENTY",
}

# ---------------------------------------------------------------------------
# State Tracking
# ---------------------------------------------------------------------------
_known_roads = set()
_known_settlements = set()
_known_cities = set()
_game_over_seen = False


def reset_parser_state():
    global _game_over_seen
    _known_roads.clear()
    _known_settlements.clear()
    _known_cities.clear()
    _game_over_seen = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ActionTuple = Tuple[int, Optional[int], Optional[str]]


def _texts_sorted(gls: dict) -> list[tuple[Optional[int], dict]]:
    result = []
    for k, v in sorted(gls.items(), key=lambda x: int(x[0])):
        t = v.get("text", {})
        if isinstance(t, dict):
            result.append((v.get("from"), t))
    return result


def _split_maritime_trades(
    given_enums: list[int], received_enums: list[int], trade_ratios: dict
) -> list[tuple]:
    """
    Greedily splits combined Colonist trades into atomic Catanatron trades
    using the player's current actual trade ratios.
    """
    given_counts = Counter(given_enums)
    trades = []

    # For each card received, find the corresponding 'given' resources that satisfy a ratio
    for recv_enum in received_enums:
        recv_res = _RESOURCE_MAP[recv_enum]
        matched = False

        # Check Wood, Brick, Sheep, Wheat, Ore in order
        for res_enum in [1, 2, 3, 4, 5]:
            if given_counts[res_enum] == 0:
                continue

            ratio = trade_ratios.get(res_enum, 4)
            if given_counts[res_enum] >= ratio:
                # Consume resources from the virtual pool
                given_counts[res_enum] -= ratio
                res_name = _RESOURCE_MAP[res_enum]

                # Catanatron expects (Res, Res, Res, Res, Target)
                # Pad with None for 2:1 or 3:1 trades
                trade_list = [res_name] * ratio + [None] * (4 - ratio) + [recv_res]
                trades.append(tuple(trade_list))
                matched = True
                break

        if not matched:
            # If this happens, it usually means the port mapping is wrong
            # and Catanatron thinks the ratio is 4:1 while Colonist used 2:1.
            print(
                f"[WARN] Could not satisfy trade for {recv_res}. Ratios used: {trade_ratios}"
            )

    return trades


# ---------------------------------------------------------------------------
# Primary parser
# ---------------------------------------------------------------------------


def parse_step(
    json_event: dict, current_player_color: int, player_ratios: dict = None
) -> Optional[List[ActionTuple]]:
    global _game_over_seen

    sc = json_event.get("stateChange", {})
    gls = sc.get("gameLogState", {})
    texts = _texts_sorted(gls)

    # Check for game over (type 45) so we can swallow post-game actions
    for _from_color, text in texts:
        if text.get("type") == 45:
            _game_over_seen = True

    parsed_actions = []

    for _from_color, text in texts:
        ttype = text.get("type")
        pcolor = text.get("playerColor")

        if ttype == 10 and pcolor == current_player_color:
            forced_roll = text.get("firstDice", 0) + text.get("secondDice", 0)
            idx = ACTIONS_ARRAY.index((AT.ROLL, None))
            parsed_actions.append((idx, forced_roll, None))

        elif ttype == 11 and pcolor == current_player_color:
            location = sc.get("mechanicRobberState", {}).get("locationTileIndex")
            if location is not None:
                catan_coord = WEB_TILE_TO_CATAN_COORD[location]
                idx = ACTIONS_ARRAY.index((AT.MOVE_ROBBER, catan_coord))
                parsed_actions.append((idx, None, None))

        elif ttype == 55 and pcolor == current_player_color:
            idx = ACTIONS_ARRAY.index((AT.DISCARD, None))
            parsed_actions.append((idx, None, None))

        elif ttype in (4, 5) and pcolor == current_player_color:
            piece_enum = text.get("pieceEnum")
            ms = sc.get("mapState", {})

            if piece_enum == 0:  # Road
                for eid_str, edata in ms.get("tileEdgeStates", {}).items():
                    if edata.get("owner") == current_player_color:
                        eid = int(eid_str)
                        if eid not in _known_roads:
                            _known_roads.add(eid)
                            catan_edge = WEB_EDGE_TO_CATAN_EDGE[eid]
                            idx = ACTIONS_ARRAY.index((AT.BUILD_ROAD, catan_edge))
                            parsed_actions.append((idx, None, None))

            elif piece_enum == 2:  # Settlement
                for cid_str, cdata in ms.get("tileCornerStates", {}).items():
                    if cdata.get("buildingType") == 1:
                        cid = int(cid_str)
                        if cid not in _known_settlements:
                            _known_settlements.add(cid)
                            catan_node = WEB_CORNER_TO_CATAN_NODE[cid]
                            idx = ACTIONS_ARRAY.index((AT.BUILD_SETTLEMENT, catan_node))
                            parsed_actions.append((idx, None, None))

            elif piece_enum in (1, 3):  # City
                for cid_str, cdata in ms.get("tileCornerStates", {}).items():
                    if cdata.get("buildingType") == 2:
                        cid = int(cid_str)
                        if cid not in _known_cities:
                            _known_cities.add(cid)
                            catan_node = WEB_CORNER_TO_CATAN_NODE[cid]
                            idx = ACTIONS_ARRAY.index((AT.BUILD_CITY, catan_node))
                            parsed_actions.append((idx, None, None))

        elif ttype == 1 and pcolor == current_player_color:
            mdc = sc.get("mechanicDevelopmentCardsState", {})
            pdata = mdc.get("players", {}).get(str(current_player_color), {})
            bought = pdata.get("developmentCardsBoughtThisTurn", [])
            dev_card = _DEVCARD_MAP.get(bought[-1]) if bought else None
            idx = ACTIONS_ARRAY.index((AT.BUY_DEVELOPMENT_CARD, None))
            parsed_actions.append((idx, None, dev_card))

        elif ttype == 20 and pcolor == current_player_color:
            card_enum = text.get("cardEnum")
            if card_enum == 11:
                idx = ACTIONS_ARRAY.index((AT.PLAY_KNIGHT_CARD, None))
                parsed_actions.append((idx, None, None))
            elif card_enum == 14:
                idx = ACTIONS_ARRAY.index((AT.PLAY_ROAD_BUILDING, None))
                parsed_actions.append((idx, None, None))
            elif card_enum == 15:
                yop_enums: list[int] = []
                for _, t2 in texts:
                    if (
                        t2.get("type") == 21
                        and t2.get("playerColor") == current_player_color
                    ):
                        yop_enums = t2.get("cardEnums", [])
                        break
                _RES_ORDER = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
                resources = tuple(
                    sorted(
                        (_RESOURCE_MAP[e] for e in yop_enums),
                        key=lambda r: _RES_ORDER.index(r),
                    )
                )
                idx = ACTIONS_ARRAY.index((AT.PLAY_YEAR_OF_PLENTY, resources))
                parsed_actions.append((idx, None, None))

        elif ttype == 116 and pcolor == current_player_color:
            given = text.get("givenCardEnums", [])
            received = text.get("receivedCardEnums", [])
            ratios = player_ratios or {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
            trade_tuples = _split_maritime_trades(given, received, ratios)
            for tt in trade_tuples:
                idx = ACTIONS_ARRAY.index((AT.MARITIME_TRADE, tt))
                parsed_actions.append((idx, None, None))

        elif ttype == 44:
            if not _game_over_seen:
                idx = ACTIONS_ARRAY.index((AT.END_TURN, None))
                parsed_actions.append((idx, None, None))

    return parsed_actions if len(parsed_actions) > 0 else None
