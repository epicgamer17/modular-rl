"""Derivation script for coordinate mappings between colonist.io and catanatron.

This script mathematically maps the Colonist (x, y, z) coordinates directly to
Catanatron's internal topology arrays, guaranteeing a flawless bijection.

Run it once to compute the mappings:
    python -m offline_data.build_mappings
"""

import json
import os
import glob
from typing import Dict, Tuple
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron"
)

from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.models.enums import NodeRef, EdgeRef

REPLAYS_DIR = os.path.join(
    os.path.dirname(__file__), "../experiments/rainbowzero/catan/replays"
)


def main():
    # 1. Initialize Catanatron Map (This builds all the internal Node IDs natively)
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)

    # 2. Read Colonist.io JSON to get web element coordinates
    replays = glob.glob(os.path.join(REPLAYS_DIR, "*.json"))
    if not replays:
        raise FileNotFoundError("No JSON replays found in REPLAYS_DIR.")

    with open(replays[0]) as f:
        data = json.load(f)["data"]

    initial_state = data.get(
        "initialState", data.get("eventHistory", {}).get("initialState", {})
    )
    ms = initial_state.get("mapState", {})

    # 3. Build Exact Tile Mapping
    WEB_TILE_TO_CATAN_COORD = {}
    for tid_str, tdata in ms.get("tileHexStates", {}).items():
        tid = int(tid_str)
        col_x, col_y = tdata["x"], tdata["y"]
        # Convert Axial to Cube
        cat_coord = (col_x, -col_x - col_y, col_y)
        WEB_TILE_TO_CATAN_COORD[tid] = cat_coord

    # 4. Build Exact Corner Mapping
    WEB_CORNER_TO_CATAN_NODE = {}
    for cid_str, cdata in ms.get("tileCornerStates", {}).items():
        cid = int(cid_str)
        col_x, col_y, col_z = cdata["x"], cdata["y"], cdata["z"]
        cat_coord = (col_x, -col_x - col_y, col_y)

        # In Colonist, z=0 is the TOP node. z=1 is the BOTTOM node.
        node_ref = NodeRef.NORTH if col_z == 0 else NodeRef.SOUTH

        # Directly extract the ID Catanatron assigned to this node
        node_id = catan_map.tiles[cat_coord].nodes[node_ref]
        WEB_CORNER_TO_CATAN_NODE[cid] = node_id

    # 5. Build Exact Edge Mapping
    WEB_EDGE_TO_CATAN_EDGE = {}
    for eid_str, edata in ms.get("tileEdgeStates", {}).items():
        eid = int(eid_str)
        col_x, col_y, col_z = edata["x"], edata["y"], edata["z"]
        # Convert Axial to Cube
        cat_coord = (col_x, -col_x - col_y, col_y)

        # Updated Mapping Logic:
        # Colonist z=0: Top-left edge -> Catanatron NORTHWEST
        if col_z == 0:
            edge_ref = EdgeRef.NORTHWEST
        # Colonist z=1: Top-right edge -> Catanatron NORTHEAST
        elif col_z == 1:
            edge_ref = EdgeRef.NORTHEAST
        # Colonist z=2: Left vertical edge -> Catanatron WEST
        elif col_z == 2:
            edge_ref = EdgeRef.WEST
        else:
            raise ValueError(f"Unknown col_z for edge: {col_z}")

        # Retrieve the edge from the tile object
        edge_id = catan_map.tiles[cat_coord].edges[edge_ref]

        # Catanatron edges MUST be sorted tuples to match your tracker's logic
        # and the WEB_EDGE_TO_CATAN_EDGE dictionary format.
        edge_tuple = tuple(sorted(edge_id))
        WEB_EDGE_TO_CATAN_EDGE[eid] = edge_tuple

    # --- Print Outputs ---
    print("WEB_CORNER_TO_CATAN_NODE: Dict[int, int] = {")
    for k in sorted(WEB_CORNER_TO_CATAN_NODE.keys()):
        print(f"    {k}: {WEB_CORNER_TO_CATAN_NODE[k]},")
    print("}")

    print("\nWEB_EDGE_TO_CATAN_EDGE: Dict[int, Tuple[int, int]] = {")
    for k in sorted(WEB_EDGE_TO_CATAN_EDGE.keys()):
        print(f"    {k}: {WEB_EDGE_TO_CATAN_EDGE[k]},")
    print("}")

    print("\nWEB_TILE_TO_CATAN_COORD: Dict[int, Tuple[int, int, int]] = {")
    for k in sorted(WEB_TILE_TO_CATAN_COORD.keys()):
        print(f"    {k}: {WEB_TILE_TO_CATAN_COORD[k]},")
    print("}")


if __name__ == "__main__":
    main()
