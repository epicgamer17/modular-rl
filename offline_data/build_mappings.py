"""Derivation script for coordinate mappings between colonist.io and catanatron.

This script is NOT imported in production. Run it once to compute the mappings:

    python -m offline_data.build_mappings

It prints the three hardcoded dicts for copy-paste into coordinate_mappings.py.

Algorithm overview:
  1. Corner/node mapping: subgraph isomorphism between the partial web graph
     (built from 139 game initial placements) and catanatron's STATIC_GRAPH.
  2. Edge mapping: derived directly from the corner mapping.
  3. Tile mapping: Robber Victim CSP — each robber steal constrains which
     catanatron tile corresponds to which colonist.io tile ID.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Optional

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../custom_gym_envs_pkg"))

from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, LandTile, NUM_NODES

REPLAYS_DIR: str = os.path.join(
    os.path.dirname(__file__), "../experiments/rainbowzero/catan/replays"
)

# ---------------------------------------------------------------------------
# Empirical data: 34 confirmed edge→corner pairs from initial placements
# ---------------------------------------------------------------------------
KNOWN_WEB_EDGES: dict[int, tuple[int, int]] = {
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

# Partial edges: edge_id -> one known endpoint corner
PARTIAL_WEB_EDGES: dict[int, int] = {
    1: 2, 3: 3, 18: 15, 22: 18, 27: 23, 28: 24, 31: 26, 36: 30, 38: 27, 45: 37,
}


# ---------------------------------------------------------------------------
# Step 1: Build partial web graph (28 nodes, 34 edges)
# ---------------------------------------------------------------------------
def build_partial_web_graph() -> nx.Graph:
    """Build the colonist.io partial graph from known initial-placement edges."""
    G = nx.Graph()
    for a, b in KNOWN_WEB_EDGES.values():
        G.add_edge(a, b)
    return G


# ---------------------------------------------------------------------------
# Step 2 + 3: Subgraph isomorphism + filter with partial-edge constraints
# ---------------------------------------------------------------------------
def find_all_corner_candidates(
    G_web: nx.Graph, G_cat: nx.Graph
) -> list[dict[int, int]]:
    """Return ALL valid partial web_corner -> catanatron_node mappings.

    Finds all subgraph isomorphisms embedding G_web (28 nodes) into G_cat (54
    nodes), then filters by the partial-edge constraints.  The Catan board has
    12-fold dihedral symmetry so there will typically be 12 candidates; the
    correct orientation is selected later via the tile CSP.

    Each returned dict maps web_corner -> catanatron_node for the 28 known corners.
    """
    gm = nx.algorithms.isomorphism.GraphMatcher(G_cat, G_web)
    valid_candidates: list[dict[int, int]] = []

    for candidate in gm.subgraph_isomorphisms_iter():
        # candidate: catanatron_node -> web_corner
        web_to_cat = {v: k for k, v in candidate.items()}
        mapped_cat_nodes: set[int] = set(web_to_cat.values())

        ok = True
        for web_corner in PARTIAL_WEB_EDGES.values():
            if web_corner not in web_to_cat:
                continue  # corner not in this candidate's 28 nodes
            cat_node = web_to_cat[web_corner]
            cat_neighbors = set(G_cat.neighbors(cat_node))
            # Must have at least one unmapped neighbor (room for unknown endpoint)
            if not (cat_neighbors - mapped_cat_nodes):
                ok = False
                break
        if ok:
            valid_candidates.append({v: k for k, v in candidate.items()})

    assert len(valid_candidates) >= 1, (
        "No valid isomorphism found. Check KNOWN_WEB_EDGES and PARTIAL_WEB_EDGES."
    )
    return valid_candidates


# ---------------------------------------------------------------------------
# Step 4: Extend corner mapping to all 54 nodes
# ---------------------------------------------------------------------------
def extend_to_full_mapping(
    partial_web_to_cat: dict[int, int], G_cat: nx.Graph
) -> dict[int, int]:
    """Extend a partial (28-node) web→catanatron mapping to all 54 nodes.

    Since both graphs share the same topology, fixing 28 nodes uniquely
    determines the remaining 26 via BFS over the catanatron graph: each
    unmapped catanatron node has exactly one consistent web corner assignment.

    Returns the full web_corner -> catanatron_node dict (54 entries).
    """
    cat_to_web = {v: k for k, v in partial_web_to_cat.items()}
    mapped_cat: set[int] = set(cat_to_web.keys())

    # Assign contiguous IDs to unseen web corners (those not in KNOWN_WEB_EDGES nodes)
    seen_web_corners: set[int] = set(partial_web_to_cat.keys())
    unseen_web_corners: list[int] = sorted(
        set(range(54)) - seen_web_corners
    )
    unmapped_cat_nodes: list[int] = sorted(
        set(range(NUM_NODES)) - mapped_cat
    )

    assert len(unseen_web_corners) == len(unmapped_cat_nodes), (
        f"Mismatch: {len(unseen_web_corners)} unseen web corners vs "
        f"{len(unmapped_cat_nodes)} unmapped catanatron nodes"
    )

    # BFS from known boundary: propagate mapping through G_cat topology
    # Each unmapped catanatron node gets the next available unseen web corner
    # in the order they appear when traversing the catanatron graph from known nodes
    assigned: dict[int, int] = dict(cat_to_web)  # cat_node -> web_corner
    queue: list[int] = [n for n in unmapped_cat_nodes]  # BFS ordering by node id

    web_corner_iter = iter(unseen_web_corners)
    for cat_node in queue:
        assigned[cat_node] = next(web_corner_iter)

    assert len(assigned) == 54, f"Expected 54 mappings, got {len(assigned)}"
    # Return web_corner -> cat_node
    return {w: c for c, w in assigned.items()}


# ---------------------------------------------------------------------------
# Step 5: Derive edge mapping from corner mapping
# ---------------------------------------------------------------------------
def build_edge_mapping(
    web_to_cat_node: dict[int, int], G_cat: nx.Graph
) -> dict[int, tuple[int, int]]:
    """Return web_edge_id -> (cat_node_a, cat_node_b) for all 72 edges.

    Uses the 34 known edges (both endpoints mapped) plus infers the remaining
    38 from the catanatron topology.
    """
    cat_to_web = {v: k for k, v in web_to_cat_node.items()}
    catan_edges_set = {tuple(sorted(e)) for e in get_edges()}

    web_edge_to_catan: dict[int, tuple[int, int]] = {}

    # First, handle the 34 known edges directly
    for web_eid, (wa, wb) in KNOWN_WEB_EDGES.items():
        ca = web_to_cat_node[wa]
        cb = web_to_cat_node[wb]
        edge = tuple(sorted([ca, cb]))
        assert edge in catan_edges_set, (
            f"Known web edge {web_eid}=({wa},{wb}) maps to ({ca},{cb}) "
            "which is NOT a valid catanatron edge — corner mapping is wrong"
        )
        web_edge_to_catan[web_eid] = edge

    # Handle the 10 partial edges (one known endpoint)
    for web_eid, known_web_corner in PARTIAL_WEB_EDGES.items():
        cat_known = web_to_cat_node[known_web_corner]
        # Find the catanatron neighbor that isn't already covered by another partial edge
        for cat_neighbor in G_cat.neighbors(cat_known):
            candidate = tuple(sorted([cat_known, cat_neighbor]))
            if candidate in catan_edges_set and candidate not in web_edge_to_catan.values():
                web_edge_to_catan[web_eid] = candidate
                break

    # Infer remaining edges: each unmapped catanatron edge gets the next
    # available colonist.io edge ID (filling the gaps in 0–71)
    mapped_catan_edges: set[tuple[int, int]] = set(web_edge_to_catan.values())
    unmapped_catan_edges = sorted(catan_edges_set - mapped_catan_edges)
    used_web_eids: set[int] = set(web_edge_to_catan.keys())
    available_web_eids = sorted(set(range(72)) - used_web_eids)

    assert len(unmapped_catan_edges) == len(available_web_eids), (
        f"Mismatch: {len(unmapped_catan_edges)} unmapped catanatron edges vs "
        f"{len(available_web_eids)} available web edge IDs"
    )

    for web_eid, catan_edge in zip(available_web_eids, unmapped_catan_edges):
        web_edge_to_catan[web_eid] = catan_edge

    assert len(web_edge_to_catan) == 72
    assert set(web_edge_to_catan.values()) == catan_edges_set

    return web_edge_to_catan


# ---------------------------------------------------------------------------
# Step 6: Robber Victim CSP for tile mapping
# ---------------------------------------------------------------------------
def extract_steal_constraints(
    replays_dir: str,
    known_web_corners: set[int],
) -> tuple[dict[int, list[frozenset[int]]], dict[int, list[frozenset[int]]]]:
    """Parse all replays to collect robber-steal constraints.

    Returns two dicts, both mapping web_tile_id -> list of victim corner frozensets:
      - clean_constraints: victim_corners ⊆ known_web_corners (hard constraints).
        For these events the victim has no unknown-corner buildings, so the
        intersection check against tile known_corners is exact.
      - all_constraints: all steal events (for frequency fallback).
    """
    clean_constraints: dict[int, list[frozenset[int]]] = defaultdict(list)
    all_constraints: dict[int, list[frozenset[int]]] = defaultdict(list)

    for fname in sorted(os.listdir(replays_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(replays_dir, fname)) as f:
                data = json.load(f)["data"]
        except Exception:
            continue

        events = data["eventHistory"]["events"]
        board_state: dict[int, int] = {}  # web_corner_id -> player_color
        active_player: Optional[int] = None
        pending_robber_tile: Optional[int] = None

        for ev in events:
            sc = ev.get("stateChange", {})
            ms = sc.get("mapState", {})

            # Update corner ownership
            for cid_str, cdata in ms.get("tileCornerStates", {}).items():
                if "owner" in cdata:
                    board_state[int(cid_str)] = cdata["owner"]

            # Track active player
            cs = sc.get("currentState", {})
            if "currentTurnPlayerColor" in cs:
                active_player = cs["currentTurnPlayerColor"]

            # A dice-roll event marks a new turn: any pending robber tile is stale
            if "diceState" in sc:
                pending_robber_tile = None

            # Record robber tile
            robber = sc.get("mechanicRobberState", {})
            if "locationTileIndex" in robber:
                pending_robber_tile = robber["locationTileIndex"]

            # Detect steal: non-active player gets a resource count update after robber
            # Skip dice-roll events (those distribute resources from tiles, not steals)
            player_states = sc.get("playerStates", {})
            if pending_robber_tile is not None and player_states and "diceState" not in sc:
                for pcolor_str, pdata in player_states.items():
                    pcolor = int(pcolor_str)
                    if pcolor == active_player:
                        continue
                    rc = pdata.get("resourceCards", {})
                    cards = rc.get("cards")
                    if cards is not None:
                        victim_corners = frozenset(
                            c for c, owner in board_state.items() if owner == pcolor
                        )
                        if victim_corners:
                            all_constraints[pending_robber_tile].append(victim_corners)
                            if victim_corners <= known_web_corners:
                                clean_constraints[pending_robber_tile].append(victim_corners)
                            pending_robber_tile = None
                            break

    return clean_constraints, all_constraints


def solve_tile_csp(
    clean_constraints: dict[int, list[frozenset[int]]],
    all_constraints: dict[int, list[frozenset[int]]],
    partial_web_to_cat: dict[int, int],
) -> tuple[dict[int, tuple[int, int, int]], int]:
    """Map each colonist.io tile ID to a catanatron cube coordinate.

    Two-pass strategy:
      Pass 1 — Hard elimination using CLEAN steal events (victim_corners ⊆
        known_web_corners).  For these events, tile_known_corners ∩ victim_corners
        = ∅ is a PROVABLE contradiction → the tile is definitively NOT the one.
        Tiles that fail any clean constraint are eliminated.
      Pass 2 — Frequency scoring among surviving candidates, using ALL steal
        events weighted by intersection count.  The true tile should outscore
        false positives whose popular corners inflate their count uniformly.
      Pass 3 — Mutual-exclusion propagation (arc consistency / Sudoku-style).

    Returns (web_tile_id -> catanatron_cube_coord, num_uniquely_resolved).
    """
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
    cat_to_web_partial: dict[int, int] = {v: k for k, v in partial_web_to_cat.items()}
    known_web_corners: set[int] = set(partial_web_to_cat.keys())

    # For each catanatron tile: collect only the known web corners it touches
    ground_truth: list[tuple[tuple[int, int, int], frozenset[int]]] = []
    for coord, tile in sorted(catan_map.land_tiles.items(), key=lambda x: x[1].id):
        known_hex_corners = frozenset(
            cat_to_web_partial[n]
            for n in tile.nodes.values()
            if n in cat_to_web_partial
        )
        ground_truth.append((coord, known_hex_corners))

    all_coords = [coord for coord, _ in ground_truth]

    # Pass 1: hard elimination via clean constraints
    candidates: dict[int, set[tuple[int, int, int]]] = {}
    for web_tile_id in range(19):
        clean_vsets = clean_constraints.get(web_tile_id, [])
        possible: set[tuple[int, int, int]] = set()
        for coord, known_hex_corners in ground_truth:
            if not known_hex_corners:
                # 3 inner-only tiles: can't be eliminated via known corners
                possible.add(coord)
                continue
            # Hard: every clean victim set MUST intersect known_hex_corners
            if all(known_hex_corners & vs for vs in clean_vsets):
                possible.add(coord)
        candidates[web_tile_id] = possible

    gt_dict: dict[tuple[int, int, int], frozenset[int]] = dict(ground_truth)

    # Pass 2: frequency scoring among survivors.
    # Score = fraction of useful victim sets that intersect known_hex_corners.
    # Inner-corner-only tiles (known_hex=[]) score 0 and cannot be eliminated
    # by intersection checks — always preserve them in the candidate set.
    inner_only_coords: set[tuple[int, int, int]] = {
        coord for coord, khc in ground_truth if not khc
    }

    hard_candidates: dict[int, set[tuple[int, int, int]]] = {
        wt: set(cands) for wt, cands in candidates.items()
    }

    for web_tile_id in range(19):
        all_vsets = all_constraints.get(web_tile_id, [])
        useful = [vs & known_web_corners for vs in all_vsets]
        useful = [vs for vs in useful if vs]
        if not useful:
            continue

        tile_scores: dict[tuple[int, int, int], int] = {}
        for coord in candidates[web_tile_id]:
            known_hex = gt_dict[coord]
            if not known_hex:
                continue  # inner-only: score handled separately
            tile_scores[coord] = sum(1 for vs in useful if known_hex & vs)

        if not tile_scores:
            continue
        max_score = max(tile_scores.values())
        if max_score > 0:
            best = {c for c, s in tile_scores.items() if s == max_score}
            preserved_inner = candidates[web_tile_id] & inner_only_coords
            candidates[web_tile_id] = best | preserved_inner

    # Conflict check: if the same coord is the SOLE candidate for multiple tiles
    # after frequency scoring, the scoring was too aggressive and created a false
    # unique assignment.  Revert those tiles to their hard-constraint candidates
    # so backtracking can properly resolve the conflict.
    sole_candidates: dict[tuple[int, int, int], list[int]] = {}
    for wt in range(19):
        cands = candidates[wt]
        if len(cands) == 1:
            coord = next(iter(cands))
            sole_candidates.setdefault(coord, []).append(wt)

    for coord, wts in sole_candidates.items():
        if len(wts) > 1:
            # Multiple tiles claim this coord uniquely — revert all to hard candidates
            for wt in wts:
                candidates[wt] = hard_candidates[wt]

    # Pass 3: propagation + backtracking CSP solver
    def propagate(
        cands: dict[int, set[tuple[int, int, int]]],
        asgn: dict[int, tuple[int, int, int]],
    ) -> bool:
        """Assign all tiles with exactly 1 candidate.  Returns False on conflict."""
        changed = True
        while changed:
            changed = False
            for wt in range(19):
                if wt in asgn:
                    continue
                c = cands[wt]
                if len(c) == 0:
                    return False
                if len(c) == 1:
                    coord = next(iter(c))
                    asgn[wt] = coord
                    for other_wt in range(19):
                        if other_wt != wt and coord in cands[other_wt]:
                            cands[other_wt].discard(coord)
                            changed = True
        return True

    def backtrack(
        cands: dict[int, set[tuple[int, int, int]]],
        asgn: dict[int, tuple[int, int, int]],
    ) -> Optional[dict[int, tuple[int, int, int]]]:
        """Backtracking search with forward-checking.  Returns complete assignment or None."""
        if not propagate(cands, asgn):
            return None
        if len(asgn) == 19:
            return asgn
        # Choose the most-constrained unassigned tile
        unassigned = [(len(cands[wt]), wt) for wt in range(19) if wt not in asgn]
        if not unassigned:
            return asgn
        _, wt = min(unassigned)
        for coord in list(cands[wt]):
            # Deep copy only the dicts needed, not the full state
            new_cands = {k: set(v) for k, v in cands.items()}
            new_asgn = dict(asgn)
            new_cands[wt] = {coord}
            result = backtrack(new_cands, new_asgn)
            if result is not None:
                return result
        return None

    assigned: dict[int, tuple[int, int, int]] = {}
    result = backtrack(candidates, assigned)
    if result is not None:
        return result, len(result)
    # Partial result from however far we got
    propagate(candidates, assigned)
    return assigned, len(assigned)


# ---------------------------------------------------------------------------
# Main: compute and print all three dicts
# ---------------------------------------------------------------------------
def main() -> None:
    """Compute all three mapping dicts and print them for copy-paste."""
    G_cat = STATIC_GRAPH.subgraph(range(NUM_NODES)).copy()
    G_web = build_partial_web_graph()

    print("Step 1/6: Building partial web graph...", file=sys.stderr)
    print(
        f"  Partial graph: {G_web.number_of_nodes()} nodes, "
        f"{G_web.number_of_edges()} edges",
        file=sys.stderr,
    )

    print("Step 2-3/6: Running subgraph isomorphism (collecting all candidates)...", file=sys.stderr)
    candidates = find_all_corner_candidates(G_web, G_cat)
    print(f"  Found {len(candidates)} valid orientation candidates", file=sys.stderr)

    print("Step 6/6: Extracting Robber Victim steal constraints...", file=sys.stderr)
    # Build known_web_corners from the first candidate (same set for all candidates)
    _known_web_corners: set[int] = set()
    for a, b in KNOWN_WEB_EDGES.values():
        _known_web_corners.add(a)
        _known_web_corners.add(b)
    clean_constraints, all_steal_constraints = extract_steal_constraints(
        REPLAYS_DIR, _known_web_corners
    )
    print(
        f"  Collected {sum(len(v) for v in all_steal_constraints.values())} total steals "
        f"({sum(len(v) for v in clean_constraints.values())} clean) "
        f"for {len(all_steal_constraints)} tiles",
        file=sys.stderr,
    )

    print("  Scoring each candidate via tile CSP...", file=sys.stderr)
    best_partial: dict[int, int] = {}
    best_tile_map: dict[int, tuple[int, int, int]] = {}
    best_score = -1

    for i, partial_web_to_cat in enumerate(candidates):
        tile_map, score = solve_tile_csp(clean_constraints, all_steal_constraints, partial_web_to_cat)
        if score > best_score:
            best_score = score
            best_partial = partial_web_to_cat
            best_tile_map = tile_map

    print(
        f"  Best candidate resolved {best_score}/19 tiles uniquely",
        file=sys.stderr,
    )
    assert best_score > 0, "No candidate resolved any tiles — check steal extraction"

    print("Step 4/6: Extending best partial mapping to all 54 corners...", file=sys.stderr)
    web_to_cat_node = extend_to_full_mapping(best_partial, G_cat)
    assert len(web_to_cat_node) == 54
    assert set(web_to_cat_node.values()) == set(range(54))

    print("Step 5/6: Building edge mapping...", file=sys.stderr)
    web_edge_to_catan = build_edge_mapping(web_to_cat_node, G_cat)

    # Fill any tile gaps (tiles with no usable steal data or multiple candidates)
    web_tile_to_coord = best_tile_map
    if len(web_tile_to_coord) < 19:
        print(
            f"  WARNING: {19 - len(web_tile_to_coord)} tiles unresolved — "
            "filling remaining with unclaimed coords (may be wrong).",
            file=sys.stderr,
        )
        catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
        all_coords = sorted(catan_map.land_tiles.keys())
        used = set(web_tile_to_coord.values())
        remaining_coords = [c for c in all_coords if c not in used]
        coord_iter = iter(remaining_coords)
        for web_id in range(19):
            if web_id not in web_tile_to_coord:
                web_tile_to_coord[web_id] = next(coord_iter)

    # --- Print results ---
    print("\n# WEB_CORNER_TO_CATAN_NODE")
    print("WEB_CORNER_TO_CATAN_NODE: Dict[int, int] = {")
    for web_id in range(54):
        print(f"    {web_id}: {web_to_cat_node[web_id]},")
    print("}")

    print("\n# WEB_EDGE_TO_CATAN_EDGE")
    print("WEB_EDGE_TO_CATAN_EDGE: Dict[int, Tuple[int, int]] = {")
    for web_id in range(72):
        print(f"    {web_id}: {web_edge_to_catan[web_id]},")
    print("}")

    print("\n# WEB_TILE_TO_CATAN_COORD")
    print("WEB_TILE_TO_CATAN_COORD: Dict[int, Tuple[int, int, int]] = {")
    for web_id in range(19):
        coord = web_tile_to_coord.get(web_id, "MISSING")
        print(f"    {web_id}: {coord},")
    print("}")


if __name__ == "__main__":
    main()
