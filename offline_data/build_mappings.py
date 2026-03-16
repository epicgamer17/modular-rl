"""Derivation script for coordinate mappings between colonist.io and catanatron.

This script procedurally generates the coordinate mappings based on the
established topological generation pattern of colonist.io:
- Tiles are ordered starting from the top-left, going counter-clockwise in rings.
- Nodes and Edges are labeled by iterating through the tiles in order,
  and walking around each tile clockwise starting from the North corner.
"""

import os
from typing import Dict, Tuple
import sys

# Ensure paths to Catanatron are correct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron"
)

from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.enums import NodeRef, EdgeRef


def main():
    # 1. Initialize Catanatron Map
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)

    # 2. Hardcode the known tile mapping which follows the counter-clockwise ring pattern
    WEB_TILE_TO_CATAN_COORD = {
        0: (0, 2, -2),
        1: (-1, 2, -1),
        2: (-2, 2, 0),
        3: (-2, 1, 1),
        4: (-2, 0, 2),
        5: (-1, -1, 2),
        6: (0, -2, 2),
        7: (1, -2, 1),
        8: (2, -2, 0),
        9: (2, -1, -1),
        10: (2, 0, -2),
        11: (1, 1, -2),
        12: (0, 1, -1),
        13: (-1, 1, 0),
        14: (-1, 0, 1),
        15: (0, -1, 1),
        16: (1, -1, 0),
        17: (1, 0, -1),
        18: (0, 0, 0),
    }

    # 3. Procedurally Generate Corners
    # Algorithm: For each tile in counter-clockwise order, go clockwise starting from NORTH.
    # Assign lowest available ID to unassigned nodes.
    WEB_CORNER_TO_CATAN_NODE = {}
    catan_node_seen = set()
    next_corner_id = 0

    clock_nodes = [
        NodeRef.NORTH,
        NodeRef.NORTHEAST,
        NodeRef.SOUTHEAST,
        NodeRef.SOUTH,
        NodeRef.SOUTHWEST,
        NodeRef.NORTHWEST,
    ]

    for tile_id in range(19):
        cat_coord = WEB_TILE_TO_CATAN_COORD[tile_id]
        for node_ref in clock_nodes:
            catan_node_id = catan_map.tiles[cat_coord].nodes[node_ref]

            # Ensure we only map land nodes (0-53) and ignore already labeled ones
            if catan_node_id < 54 and catan_node_id not in catan_node_seen:
                WEB_CORNER_TO_CATAN_NODE[next_corner_id] = catan_node_id
                catan_node_seen.add(catan_node_id)
                next_corner_id += 1

    # 4. Procedurally Generate Edges
    # Algorithm: Same as corners. Clockwise starting from the edge originating at the NORTH corner.
    WEB_EDGE_TO_CATAN_EDGE = {}
    catan_edge_seen = set()
    next_edge_id = 0

    clock_edges = [
        EdgeRef.NORTHEAST,  # North to NorthEast
        EdgeRef.EAST,  # NorthEast to SouthEast
        EdgeRef.SOUTHEAST,  # SouthEast to South
        EdgeRef.SOUTHWEST,  # South to SouthWest
        EdgeRef.WEST,  # SouthWest to NorthWest
        EdgeRef.NORTHWEST,  # NorthWest to North
    ]

    for tile_id in range(19):
        cat_coord = WEB_TILE_TO_CATAN_COORD[tile_id]
        for edge_ref in clock_edges:
            catan_edge_id = catan_map.tiles[cat_coord].edges[edge_ref]

            # Ensure both endpoints are land nodes to filter out ports
            if all(n < 54 for n in catan_edge_id):
                edge_tuple = tuple(sorted(catan_edge_id))

                # Ignore already labeled edges
                if edge_tuple not in catan_edge_seen:
                    WEB_EDGE_TO_CATAN_EDGE[next_edge_id] = edge_tuple
                    catan_edge_seen.add(edge_tuple)
                    next_edge_id += 1

    # --- Print Outputs ---
    print("WEB_CORNER_TO_CATAN_NODE: Dict[int, int] = {")
    for k in range(len(WEB_CORNER_TO_CATAN_NODE)):
        print(f"    {k}: {WEB_CORNER_TO_CATAN_NODE[k]},")
    print("}")

    print("\nWEB_EDGE_TO_CATAN_EDGE: Dict[int, Tuple[int, int]] = {")
    for k in range(len(WEB_EDGE_TO_CATAN_EDGE)):
        print(f"    {k}: {WEB_EDGE_TO_CATAN_EDGE[k]},")
    print("}")

    print("\nWEB_TILE_TO_CATAN_COORD: Dict[int, Tuple[int, int, int]] = {")
    for k in range(19):
        print(f"    {k}: {WEB_TILE_TO_CATAN_COORD[k]},")
    print("}")


if __name__ == "__main__":
    main()

# confirmed_edges = {
#     0: (45, 46),  #
#     1: (46, 19),
#     2: (19, 21),  #
#     3: (21, 43),  #
#     4: (43, 47),  #
#     5: (47, 45),
#     6: (21, 16),
#     7: (16, 18),  #
#     8: (18, 40),
#     9: (40, 44),
#     10: (44, 43),
#     21: (13, 14),  #
#     22: (13, 34),  #
#     62: (5, 16),  #
#     66: (3, 12),  #
#     68: (2, 3),  #
# }
