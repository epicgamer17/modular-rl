"""Mapping between colonist.io game coordinate IDs and catanatron internal IDs.

Colonist.io uses integer IDs for board elements:
  - Corners/nodes: 0–53
  - Edges/roads:   0–71
  - Hex tiles:     0–18

Catanatron uses a different integer ordering for nodes (0–53), sorted-tuple
edges, and cube coordinates (x, y, z) with x+y+z=0 for tiles.

These mappings were derived empirically via:
  1. Graph isomorphism from 139 colonist.io game replays (initial-placement
     adjacency constraints) to find the unique corner/node bijection.
  2. Robber-Victim CSP: robber move + victim corner sets from steal events
     uniquely identify each tile's catanatron cube coordinate.

Run `python -m offline_data.build_mappings` to regenerate from raw replays.
"""

from typing import Dict, Tuple

NUM_WEB_CORNERS: int = 54
NUM_WEB_EDGES: int = 72
NUM_WEB_TILES: int = 19
NUM_CATAN_NODES: int = 54

# ---------------------------------------------------------------------------
# colonist.io corner ID (0–53) → catanatron node ID (0–53)
# ---------------------------------------------------------------------------
WEB_CORNER_TO_CATAN_NODE: Dict[int, int] = {
    0: 24,
    1: 25,
    2: 19,
    3: 21,
    4: 26,
    5: 27,
    6: 16,
    7: 18,
    8: 40,
    9: 28,
    10: 17,
    11: 39,
    12: 29,
    13: 30,
    14: 15,
    15: 14,
    16: 31,
    17: 32,
    18: 13,
    19: 33,
    20: 34,
    21: 35,
    22: 12,
    23: 11,
    24: 36,
    25: 37,
    26: 10,
    27: 38,
    28: 41,
    29: 42,
    30: 8,
    31: 43,
    32: 44,
    33: 9,
    34: 45,
    35: 46,
    36: 47,
    37: 7,
    38: 52,
    39: 48,
    40: 6,
    41: 23,
    42: 50,
    43: 51,
    44: 22,
    45: 49,
    46: 53,
    47: 20,
    48: 0,
    49: 5,
    50: 4,
    51: 3,
    52: 2,
    53: 1,
}

# ---------------------------------------------------------------------------
# colonist.io edge ID (0–71) → catanatron edge as sorted (node_a, node_b)
# ---------------------------------------------------------------------------
WEB_EDGE_TO_CATAN_EDGE: Dict[int, Tuple[int, int]] = {
    0: (24, 25),
    1: (19, 46),
    2: (19, 21),
    3: (21, 43),
    4: (24, 53),
    5: (25, 26),
    6: (16, 21),
    7: (16, 18),
    8: (18, 40),
    9: (26, 27),
    10: (27, 28),
    11: (17, 18),
    12: (17, 39),
    13: (28, 29),
    14: (29, 30),
    15: (30, 31),
    16: (15, 17),
    17: (14, 15),
    18: (14, 37),
    19: (31, 32),
    20: (32, 33),
    21: (13, 14),
    22: (13, 34),
    23: (33, 34),
    24: (34, 35),
    25: (36, 37),
    26: (11, 12),
    27: (11, 32),
    28: (35, 36),
    29: (38, 39),
    30: (12, 13),
    31: (10, 29),
    32: (39, 41),
    33: (40, 42),
    34: (40, 44),
    35: (10, 11),
    36: (8, 27),
    37: (41, 42),
    38: (37, 38),
    39: (9, 10),
    40: (8, 9),
    41: (43, 44),
    42: (43, 47),
    43: (45, 46),
    44: (7, 8),
    45: (7, 24),
    46: (45, 47),
    47: (46, 48),
    48: (6, 7),
    49: (6, 23),
    50: (23, 52),
    51: (48, 49),
    52: (49, 50),
    53: (22, 23),
    54: (22, 49),
    55: (50, 51),
    56: (51, 52),
    57: (20, 22),
    58: (19, 20),
    59: (52, 53),
    60: (0, 20),
    61: (0, 5),
    62: (5, 16),
    63: (4, 5),
    64: (4, 15),
    65: (3, 4),
    66: (3, 12),
    67: (2, 9),
    68: (2, 3),
    69: (1, 2),
    70: (1, 6),
    71: (0, 1),
}

WEB_TILE_TO_CATAN_COORD: Dict[int, Tuple[int, int, int]] = {
    0: (0, 2, -2),   1: (-1, 2, -1),  2: (-2, 2, 0),
    3: (-2, 1, 1),   4: (-2, 0, 2),   5: (-1, -1, 2),
    6: (0, -2, 2),   7: (1, -2, 1),   8: (2, -2, 0),
    9: (2, -1, -1),  10: (2, 0, -2),  11: (1, 1, -2),
    12: (0, 1, -1),  13: (-1, 1, 0),  14: (-1, 0, 1),
    15: (0, -1, 1),  16: (1, -1, 0),  17: (1, 0, -1),
    18: (0, 0, 0)
}

# ---------------------------------------------------------------------------
# Inverse mappings (derived at module load)
# ---------------------------------------------------------------------------
CATAN_NODE_TO_WEB_CORNER: Dict[int, int] = {
    v: k for k, v in WEB_CORNER_TO_CATAN_NODE.items()
}
CATAN_EDGE_TO_WEB_EDGE: Dict[Tuple[int, int], int] = {
    v: k for k, v in WEB_EDGE_TO_CATAN_EDGE.items()
}
CATAN_COORD_TO_WEB_TILE: Dict[Tuple[int, int, int], int] = {
    v: k for k, v in WEB_TILE_TO_CATAN_COORD.items()
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def web_corner_to_catan_node(web_corner_id: int) -> int:
    """Convert colonist.io corner ID (0–53) to catanatron node ID (0–53)."""
    assert web_corner_id in WEB_CORNER_TO_CATAN_NODE, (
        f"Invalid colonist corner ID: {web_corner_id}. Expected 0–{NUM_WEB_CORNERS - 1}."
    )
    return WEB_CORNER_TO_CATAN_NODE[web_corner_id]


def web_edge_to_catan_edge(web_edge_id: int) -> Tuple[int, int]:
    """Convert colonist.io edge ID (0–71) to catanatron edge (sorted node-id pair)."""
    assert web_edge_id in WEB_EDGE_TO_CATAN_EDGE, (
        f"Invalid colonist edge ID: {web_edge_id}. Expected 0–{NUM_WEB_EDGES - 1}."
    )
    return WEB_EDGE_TO_CATAN_EDGE[web_edge_id]


def web_tile_to_catan_coord(web_tile_id: int) -> Tuple[int, int, int]:
    """Convert colonist.io tile ID (0–18) to catanatron cube coordinate (x, y, z)."""
    assert web_tile_id in WEB_TILE_TO_CATAN_COORD, (
        f"Invalid colonist tile ID: {web_tile_id}. Expected 0–{NUM_WEB_TILES - 1}."
    )
    return WEB_TILE_TO_CATAN_COORD[web_tile_id]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_mappings() -> None:
    """Assert full bijection and consistency with catanatron's graph structure."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from catanatron.models.board import get_edges
    from catanatron.models.map import BASE_MAP_TEMPLATE, LandTile

    # Corner bijection
    assert len(WEB_CORNER_TO_CATAN_NODE) == NUM_WEB_CORNERS, "Missing corner mappings"
    assert set(WEB_CORNER_TO_CATAN_NODE.keys()) == set(range(NUM_WEB_CORNERS))
    assert set(WEB_CORNER_TO_CATAN_NODE.values()) == set(range(NUM_CATAN_NODES))

    # Edge bijection + values are valid catanatron edges
    catan_edges = {tuple(sorted(e)) for e in get_edges()}
    assert len(WEB_EDGE_TO_CATAN_EDGE) == NUM_WEB_EDGES, "Missing edge mappings"
    assert set(WEB_EDGE_TO_CATAN_EDGE.keys()) == set(range(NUM_WEB_EDGES))
    assert set(WEB_EDGE_TO_CATAN_EDGE.values()) == catan_edges, (
        "Edge values don't match catanatron"
    )

    # Tile bijection + values are valid land tile cube coords
    catan_land_coords = {
        coord for coord, t in BASE_MAP_TEMPLATE.topology.items() if t == LandTile
    }
    assert len(WEB_TILE_TO_CATAN_COORD) == NUM_WEB_TILES, "Missing tile mappings"
    assert set(WEB_TILE_TO_CATAN_COORD.keys()) == set(range(NUM_WEB_TILES))
    assert set(WEB_TILE_TO_CATAN_COORD.values()) == catan_land_coords, (
        "Tile coords don't match BASE_MAP_TEMPLATE land tiles"
    )

    # Known empirical assertions from initial-placement data (34 confirmed edges):
    # Each KNOWN_WEB_EDGES pair must map to a valid catanatron edge whose endpoints
    # match the corner mapping.
    KNOWN_WEB_EDGES = {
        2: (2, 3), 6: (3, 6), 7: (6, 7), 8: (7, 8), 11: (7, 10),
        12: (10, 11), 16: (10, 14), 17: (14, 15), 21: (15, 18),
        26: (22, 23), 30: (18, 22), 35: (23, 26), 39: (26, 33),
        40: (30, 33), 44: (30, 37), 48: (37, 40), 49: (40, 41),
        50: (38, 41), 53: (41, 44), 54: (44, 45), 57: (44, 47),
        58: (2, 47), 60: (47, 48), 61: (48, 49), 62: (6, 49),
        63: (49, 50), 64: (14, 50), 65: (50, 51), 66: (22, 51),
        67: (33, 52), 68: (51, 52), 69: (52, 53), 70: (40, 53),
        71: (48, 53),
    }
    for web_edge_id, (wa, wb) in KNOWN_WEB_EDGES.items():
        ca = web_corner_to_catan_node(wa)
        cb = web_corner_to_catan_node(wb)
        expected_edge = tuple(sorted([ca, cb]))
        actual_edge = web_edge_to_catan_edge(web_edge_id)
        assert actual_edge == expected_edge, (
            f"Web edge {web_edge_id}=({wa},{wb}) should map to {expected_edge}, "
            f"got {actual_edge}"
        )
        assert expected_edge in catan_edges, (
            f"Edge {expected_edge} is not a valid catanatron edge"
        )

    print("All coordinate mapping checks passed.")


if __name__ == "__main__":
    _verify_mappings()
