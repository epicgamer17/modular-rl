"""Parse colonist.io game log events into catanatron action indices and RNG seeds.

Each colonist.io replay is a sequence of delta events. This module converts each
event into the tuple (action_idx, forced_roll, forced_dev_card) consumed by the
catanatron custom PettingZoo environment:

  action_idx       – integer index into ACTIONS_ARRAY (0–289); None if not parseable
  forced_roll      – int dice sum (2–12) when the event was a dice roll; else None
  forced_dev_card  – str dev card name ('KNIGHT', 'YEAR_OF_PLENTY', etc.)
                     when a dev card was purchased; else None

Colonist.io game log text types handled:
  type 1   → BUY_DEVELOPMENT_CARD
  type 4   → BUILD_ROAD / BUILD_SETTLEMENT (initial placement)
  type 5   → BUILD_ROAD / BUILD_SETTLEMENT / BUILD_CITY (purchased)
  type 10  → ROLL
  type 11  → MOVE_ROBBER
  type 20  → PLAY_KNIGHT_CARD / PLAY_ROAD_BUILDING / PLAY_YEAR_OF_PLENTY /
             PLAY_MONOPOLY
  type 44  → END_TURN
  type 55  → DISCARD
  type 116 → MARITIME_TRADE (may batch multiple trades in one event)

Colonist.io resource enum mapping (1-indexed):
  1=WOOD, 2=BRICK, 3=SHEEP, 4=WHEAT, 5=ORE

Colonist.io dev card enum mapping:
  11=KNIGHT, 12=MONOPOLY, 13=VICTORY_POINT, 14=ROAD_BUILDING, 15=YEAR_OF_PLENTY
  (10=hidden/bank)
"""

import sys
import os
from typing import Optional, List, Tuple

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

# ---------------------------------------------------------------------------
# Enum mappings
# ---------------------------------------------------------------------------

# colonist.io resource enum (1-indexed) → catanatron resource string
_RESOURCE_MAP: dict[int, str] = {
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}

# colonist.io dev card enum → catanatron dev card name
# 10 = hidden (bank), 13 = VICTORY_POINT (not playable, yields None action)
_DEVCARD_MAP: dict[int, str] = {
    11: "KNIGHT",
    12: "MONOPOLY",
    13: "VICTORY_POINT",
    14: "ROAD_BUILDING",
    15: "YEAR_OF_PLENTY",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ActionTuple = Tuple[int, Optional[int], Optional[str]]


def _texts_sorted(gls: dict) -> list[tuple[Optional[int], dict]]:
    """Return (from_color, text_dict) pairs sorted by log entry key."""
    result = []
    for k, v in sorted(gls.items(), key=lambda x: int(x[0])):
        t = v.get("text", {})
        if isinstance(t, dict):
            result.append((v.get("from"), t))
    return result


def _split_maritime_trades(
    given_enums: list[int], received_enums: list[int]
) -> list[tuple]:
    """Decompose a (possibly batched) colonist.io maritime trade into individual
    catanatron MARITIME_TRADE tuples.

    Each sub-trade uses cards of exactly one given resource type. The n:1 ratio
    is encoded as:
      4:1  → (res, res, res, res, recv)
      3:1  → (res, res, res, None, recv)
      2:1  → (res, res, None, None, recv)

    Batched events (len(received) > 1) are split by grouping given cards by
    resource and pairing groups with received resources in order of appearance.
    When all given are the same resource, the cards are split evenly.
    """
    num_received = len(received_enums)
    assert num_received >= 1

    # Group given cards by resource, preserving insertion order.
    groups: list[tuple[int, int]] = []  # (resource_enum, count)
    seen: dict[int, int] = {}  # resource_enum → index in groups
    for r in given_enums:
        if r not in seen:
            seen[r] = len(groups)
            groups.append((r, 0))
        idx = seen[r]
        groups[idx] = (groups[idx][0], groups[idx][1] + 1)

    trades: list[tuple] = []

    if len(groups) == num_received:
        # One distinct given resource per received resource.
        for (res_enum, count), recv_enum in zip(groups, received_enums):
            res = _RESOURCE_MAP[res_enum]
            recv = _RESOURCE_MAP[recv_enum]
            trades.extend(_build_trade_tuples(res, count, recv))

    elif len(groups) == 1:
        # All given are the same resource; split evenly across received.
        res_enum, total_count = groups[0]
        res = _RESOURCE_MAP[res_enum]
        per_trade, remainder = divmod(total_count, num_received)
        for i, recv_enum in enumerate(received_enums):
            n = per_trade + (1 if i < remainder else 0)
            recv = _RESOURCE_MAP[recv_enum]
            trades.extend(_build_trade_tuples(res, n, recv))

    else:
        # Mixed resources with fewer groups than received — fall back to greedy:
        # assign groups in order, splitting the last group if needed.
        gi = 0
        for recv_enum in received_enums:
            if gi < len(groups):
                res_enum, count = groups[gi]
                gi += 1
            else:
                # Reuse last group (shouldn't happen with well-formed data).
                res_enum, count = groups[-1]
            res = _RESOURCE_MAP[res_enum]
            recv = _RESOURCE_MAP[recv_enum]
            trades.extend(_build_trade_tuples(res, count, recv))

    return trades


def _build_trade_tuples(res: str, n: int, recv: str) -> list[tuple]:
    """Build one or more MARITIME_TRADE value tuples for a single received resource.

    Standard ratios (2–4) produce one tuple.  n > 4 (e.g. two 3:1 trades batched
    into one event by colonist.io) is decomposed greedily into valid sub-trades of
    size 4, 3, or 2 from largest to smallest.

    n=4: [(res, res, res, res, recv)]
    n=3: [(res, res, res, None, recv)]
    n=2: [(res, res, None, None, recv)]
    n=6: [(res×4, recv), (res×2, recv)]  – greedy split
    """
    result: list[tuple] = []
    remaining = n
    for size in (4, 3, 2):
        while remaining >= size:
            pad = [None] * (4 - size)
            result.append(tuple([res] * size + pad + [recv]))
            remaining -= size
    if remaining > 0:
        # Leftover 1 card — skip (invalid trade fragment).
        pass
    return result


# ---------------------------------------------------------------------------
# Primary parser
# ---------------------------------------------------------------------------


def parse_step(
    json_event: dict, current_player_color: int
) -> Optional[List[ActionTuple]]:
    """Parse one colonist.io event into a list of (action_idx, forced_roll, forced_dev_card).

    Returns a list because colonist.io occasionally batches multiple maritime
    trades into a single event. In all other cases the list has exactly one
    element. Returns None if the event contains no parseable action for
    current_player_color.

    Args:
        json_event: One entry from data.eventHistory.events.
        current_player_color: Colonist.io player color integer (e.g. 1 or 5).
    """
    sc = json_event.get("stateChange", {})
    gls = sc.get("gameLogState", {})
    texts = _texts_sorted(gls)

    for _from_color, text in texts:
        ttype = text.get("type")
        pcolor = text.get("playerColor")

        # ------------------------------------------------------------------ ROLL
        if ttype == 10 and pcolor == current_player_color:
            # diceState sometimes omits one die; use gameLogState text values.
            forced_roll = text.get("firstDice", 0) + text.get("secondDice", 0)
            idx = ACTIONS_ARRAY.index((AT.ROLL, None))
            return [(idx, forced_roll, None)]

        # ------------------------------------------------------------ MOVE_ROBBER
        if ttype == 11 and pcolor == current_player_color:
            location = sc.get("mechanicRobberState", {}).get("locationTileIndex")
            catan_coord = WEB_TILE_TO_CATAN_COORD[location]
            idx = ACTIONS_ARRAY.index((AT.MOVE_ROBBER, catan_coord))
            return [(idx, None, None)]

        # --------------------------------------------------------------- DISCARD
        if ttype == 55 and pcolor == current_player_color:
            idx = ACTIONS_ARRAY.index((AT.DISCARD, None))
            return [(idx, None, None)]

        # ------------------------------------------------- BUILD (initial or paid)
        if ttype in (4, 5) and pcolor == current_player_color:
            piece_enum = text.get("pieceEnum")
            ms = sc.get("mapState", {})

            if piece_enum == 0:  # Road
                for eid_str, edata in ms.get("tileEdgeStates", {}).items():
                    if edata.get("owner") == current_player_color:
                        catan_edge = WEB_EDGE_TO_CATAN_EDGE[int(eid_str)]
                        idx = ACTIONS_ARRAY.index((AT.BUILD_ROAD, catan_edge))
                        return [(idx, None, None)]

            elif piece_enum == 2:  # Settlement
                for cid_str, cdata in ms.get("tileCornerStates", {}).items():
                    if (
                        cdata.get("owner") == current_player_color
                        and cdata.get("buildingType") == 1
                    ):
                        catan_node = WEB_CORNER_TO_CATAN_NODE[int(cid_str)]
                        idx = ACTIONS_ARRAY.index((AT.BUILD_SETTLEMENT, catan_node))
                        return [(idx, None, None)]

            elif piece_enum in (1, 3):  # City (pieceEnum 1=city, 3=city+VP)
                for cid_str, cdata in ms.get("tileCornerStates", {}).items():
                    if (
                        cdata.get("owner") == current_player_color
                        and cdata.get("buildingType") == 2
                    ):
                        catan_node = WEB_CORNER_TO_CATAN_NODE[int(cid_str)]
                        idx = ACTIONS_ARRAY.index((AT.BUILD_CITY, catan_node))
                        return [(idx, None, None)]

        # ------------------------------------------------- BUY_DEVELOPMENT_CARD
        if ttype == 1 and pcolor == current_player_color:
            mdc = sc.get("mechanicDevelopmentCardsState", {})
            pdata = mdc.get("players", {}).get(str(current_player_color), {})
            bought = pdata.get("developmentCardsBoughtThisTurn", [])
            dev_card = _DEVCARD_MAP.get(bought[-1]) if bought else None
            idx = ACTIONS_ARRAY.index((AT.BUY_DEVELOPMENT_CARD, None))
            return [(idx, None, dev_card)]

        # -------------------------------------------------------- PLAY DEV CARD
        if ttype == 20 and pcolor == current_player_color:
            card_enum = text.get("cardEnum")

            if card_enum == 11:  # KNIGHT
                idx = ACTIONS_ARRAY.index((AT.PLAY_KNIGHT_CARD, None))
                return [(idx, None, None)]

            if card_enum == 14:  # ROAD_BUILDING
                idx = ACTIONS_ARRAY.index((AT.PLAY_ROAD_BUILDING, None))
                return [(idx, None, None)]

            if card_enum == 15:  # YEAR_OF_PLENTY
                # The chosen resources appear in type 21 in the same event.
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
                return [(idx, None, None)]

            if card_enum == 12:  # MONOPOLY
                # Colonist.io encodes monopoly resource in type 22 (not seen in
                # this replay). For now, return None to mark as unhandled.
                # TODO: identify colonist.io type for monopoly resource selection.
                return None

            # card_enum 13 = VICTORY_POINT — not in ACTIONS_ARRAY, skip.
            return None

        # ---------------------------------------------------------- MARITIME_TRADE
        if ttype == 116 and pcolor == current_player_color:
            given = text.get("givenCardEnums", [])
            received = text.get("receivedCardEnums", [])
            trade_tuples = _split_maritime_trades(given, received)
            result: List[ActionTuple] = []
            for tt in trade_tuples:
                idx = ACTIONS_ARRAY.index((AT.MARITIME_TRADE, tt))
                result.append((idx, None, None))
            return result if result else None

        # --------------------------------------------------------------- END_TURN
        if ttype == 44:
            # type 44 has no playerColor; caller is responsible for only invoking
            # parse_step with END_TURN events during current_player_color's turn.
            idx = ACTIONS_ARRAY.index((AT.END_TURN, None))
            return [(idx, None, None)]

    return None


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
        data = json.load(f)

    d = data["data"]
    events = d["eventHistory"]["events"]
    play_order: list[int] = d["playOrder"]

    # Track whose turn it is by watching currentState.currentTurnPlayerColor
    # and the play order for initial placement.
    current_turn_color: Optional[int] = play_order[0]  # first player places first

    # The initial placements in colonist.io alternate: P1 place, P2 place, P2 place, P1 place
    # The gameLogState text always carries playerColor for placements, so we rely on that.

    parsed_count = 0
    TARGET = 30

    print(
        f"{'#':<4}  {'Event':<6}  {'gameLogState (text types)':<55}  {'Parsed tuple'}"
    )
    print("-" * 110)

    for ev_idx, event in enumerate(events):
        if parsed_count >= TARGET:
            break

        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        # Determine the acting player from text (reliable for all action types
        # except END_TURN which has no playerColor).  Fall back to current_turn_color.
        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        # Summarise gameLogState for display.
        gls_summary_parts = []
        for _, t in texts:
            ttype = t.get("type")
            pc = t.get("playerColor", "?")
            extra = ""
            if ttype == 10:
                extra = f" {t.get('firstDice')}+{t.get('secondDice')}"
            elif ttype == 20:
                extra = f" cardEnum={t.get('cardEnum')}"
            elif ttype in (4, 5):
                piece_names = {0: "road", 1: "city", 2: "sett", 3: "city+"}
                extra = f" {piece_names.get(t.get('pieceEnum'), '?')}"
            elif ttype == 116:
                extra = f" {t.get('givenCardEnums')}→{t.get('receivedCardEnums')}"
            gls_summary_parts.append(f"t{ttype}(p{pc}){extra}")
        gls_summary = ", ".join(gls_summary_parts)

        # Update current_turn_color AFTER determining acting_player so END_TURN
        # events in the same stateChange as a turn transition use the old value.
        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        result = parse_step(event, acting_player)
        if result is None:
            continue

        for action_tuple in result:
            action_idx, forced_roll, forced_dev_card = action_tuple
            action_type = ACTIONS_ARRAY[action_idx][0].name
            print(
                f"{parsed_count:<4}  ev{ev_idx:<4}  {gls_summary:<55}  "
                f"({action_idx}, {str(forced_roll):<4}, {str(forced_dev_card):<20})  [{action_type}]"
            )
            parsed_count += 1
            if parsed_count >= TARGET:
                break
