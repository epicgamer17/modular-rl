"""Visualize a colonist.io game replay as an MP4 video.

Renders the board state after each action using catanatron's axial coordinate
system. Tiles show resource type and dice number. Roads are colored by player.
Buildings show settlement (circle) vs city (square). A text panel shows the
current turn number, last roll, and each player's resource count.

Usage::

    python -m offline_data.visualize_trajectory  # writes trajectory_verification.mp4
    python -m offline_data.visualize_trajectory path/to/replay.json output.mp4
"""

import json
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0,
    "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron",
)

import imageio.v2 as imageio
from catanatron.gym.board_tensor_features import get_axial_node_edge_maps
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.models.player import Color

from offline_data.json_parser import _texts_sorted, parse_step
from offline_data.path_a_simulator import GodModeStepper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESOURCE_FACE_COLORS: dict[Optional[str], str] = {
    "WOOD":  "#3a7d2c",
    "BRICK": "#b84c22",
    "SHEEP": "#8bc34a",
    "WHEAT": "#f5c518",
    "ORE":   "#7b8c9e",
    None:    "#d4b483",  # desert
}

_RESOURCE_LABELS: dict[Optional[str], str] = {
    "WOOD":  "W",
    "BRICK": "Br",
    "SHEEP": "Sh",
    "WHEAT": "Wh",
    "ORE":   "Or",
    None:    "Des",
}

_PLAYER_MCOLORS: dict[Color, str] = {
    Color.RED:    "red",
    Color.BLUE:   "dodgerblue",
    Color.ORANGE: "darkorange",
    Color.WHITE:  "white",
}

_PLAYER_LABELS: dict[Color, str] = {
    Color.RED:    "RED",
    Color.BLUE:   "BLUE",
    Color.ORANGE: "ORANGE",
    Color.WHITE:  "WHITE",
}

# Cached axial maps (NodeId → (col, row), edge-pair → (col, row))
_NODE_MAP: Optional[dict] = None
_EDGE_MAP: Optional[dict] = None


def _get_axial_maps() -> tuple[dict, dict]:
    global _NODE_MAP, _EDGE_MAP
    if _NODE_MAP is None:
        _NODE_MAP, _EDGE_MAP = get_axial_node_edge_maps()
    return _NODE_MAP, _EDGE_MAP


# ---------------------------------------------------------------------------
# Board drawing
# ---------------------------------------------------------------------------

def draw_board(
    state,
    ax: plt.Axes,
    last_roll: Optional[int] = None,
    step: int = 0,
    action_desc: str = "",
) -> None:
    """Draw the current Catan board state onto *ax*.

    Args:
        state:       catanatron game state (game.state).
        ax:          Matplotlib Axes to draw on.
        last_roll:   Dice sum from the last ROLL action, or None.
        step:        Action step index shown in the title.
        action_desc: Short description of the last action for the title.
    """
    node_map, _ = _get_axial_maps()

    ax.clear()
    ax.set_aspect("equal")
    # Grid spans col 0..20, row 0..10 for the default board.
    ax.set_xlim(-1, 23)
    ax.set_ylim(-1, 13)
    ax.invert_yaxis()
    ax.axis("off")

    board = state.board
    game_map = board.map

    # ------------------------------------------------------------------
    # Tiles
    # ------------------------------------------------------------------
    for coord, tile in game_map.land_tiles.items():
        tile_nodes = list(tile.nodes.values())
        positions = [node_map[n] for n in tile_nodes if n in node_map]
        if len(positions) < 3:
            continue

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        resource: Optional[str] = tile.resource
        face_color = _RESOURCE_FACE_COLORS.get(resource, "#d4b483")
        poly = plt.Polygon(
            list(zip(xs, ys)),
            closed=True,
            facecolor=face_color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            zorder=0,
        )
        ax.add_patch(poly)

        # Resource abbreviation
        label = _RESOURCE_LABELS.get(resource, "?")
        ax.text(cx, cy - 0.45, label, ha="center", va="center",
                fontsize=5, fontweight="bold", color="black", zorder=1)

        # Dice number (red for 6 and 8)
        if tile.number is not None:
            num_color = "darkred" if tile.number in (6, 8) else "black"
            ax.text(cx, cy + 0.45, str(tile.number), ha="center", va="center",
                    fontsize=6, color=num_color, fontweight="bold", zorder=1)

        # Robber marker
        if board.robber_coordinate == coord:
            ax.text(cx, cy, "X", ha="center", va="center",
                    fontsize=9, color="red", fontweight="bold", zorder=2)

    # ------------------------------------------------------------------
    # Roads
    # ------------------------------------------------------------------
    drawn_roads: set[tuple[int, int]] = set()
    for edge, color in board.roads.items():
        canonical: tuple[int, int] = (min(edge), max(edge))
        if canonical in drawn_roads:
            continue
        drawn_roads.add(canonical)

        a, b = canonical
        if a not in node_map or b not in node_map:
            continue

        ax_pos, ay_pos = node_map[a]
        bx_pos, by_pos = node_map[b]
        mcolor = _PLAYER_MCOLORS.get(color, "gray")
        road_edge = "black" if mcolor == "white" else mcolor
        ax.plot(
            [ax_pos, bx_pos], [ay_pos, by_pos],
            color=road_edge, linewidth=2.5, solid_capstyle="round", zorder=2,
        )

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------
    for node_id, (color, building_type) in board.buildings.items():
        if node_id not in node_map:
            continue
        x, y = node_map[node_id]
        mcolor = _PLAYER_MCOLORS.get(color, "gray")
        ec = "black"

        if building_type == SETTLEMENT:
            circle = mpatches.Circle(
                (x, y), radius=0.38,
                facecolor=mcolor, edgecolor=ec, linewidth=0.8, zorder=3,
            )
            ax.add_patch(circle)
        else:  # CITY
            rect = mpatches.FancyBboxPatch(
                (x - 0.4, y - 0.4), 0.8, 0.8,
                boxstyle="square,pad=0",
                facecolor=mcolor, edgecolor=ec, linewidth=0.8, zorder=3,
            )
            ax.add_patch(rect)

    # ------------------------------------------------------------------
    # Info panel: title + per-player resource counts
    # ------------------------------------------------------------------
    title = f"Step {step}"
    if last_roll is not None:
        title += f"  |  Roll: {last_roll}"
    if action_desc:
        title += f"  |  {action_desc}"
    ax.set_title(title, fontsize=7, pad=3)

    # Per-player resource count (bottom of axes)
    parts = []
    for color in Color:
        if color not in state.color_to_index:
            continue
        idx = state.color_to_index[color]
        total = int(sum(state.player_state.get(f"P{idx}_{r}_IN_HAND", 0) for r in RESOURCES))
        vp = int(state.player_state.get(f"P{idx}_ACTUAL_VICTORY_POINTS", 0))
        parts.append(f"{_PLAYER_LABELS[color]}:{total}res {vp}vp")

    ax.text(
        11, 12.5, "   ".join(parts),
        ha="center", va="bottom", fontsize=5,
        transform=ax.transData, zorder=4,
    )


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def generate_video(
    json_path: str,
    output_mp4: str,
    fps: int = 3,
    max_steps: Optional[int] = None,
) -> None:
    """Generate an MP4 video of the game trajectory.

    For each action in the replay, the board state is rendered and added
    as a frame.  Frames are written at *fps* frames-per-second.

    Args:
        json_path:   Path to the colonist.io replay JSON.
        output_mp4:  Path for the output MP4 file.
        fps:         Frames per second (default 3).
        max_steps:   If set, stop after this many action steps.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    d = data["data"]
    events = d["eventHistory"]["events"]
    play_order: list[int] = d["playOrder"]

    stepper = GodModeStepper()
    current_turn_color: int = play_order[0]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    writer = imageio.get_writer(output_mp4, fps=fps, macro_block_size=None)

    step_count = 0
    last_roll: Optional[int] = None

    # Render the initial empty board as frame 0.
    game = stepper.env.unwrapped.game
    draw_board(game.state, ax, last_roll=None, step=0, action_desc="(initial)")
    fig.tight_layout(pad=0.3)
    frame = _fig_to_rgb(fig)
    writer.append_data(frame)

    for ev_idx, event in enumerate(events):
        if max_steps is not None and step_count >= max_steps:
            break

        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        # Determine acting player from text (same logic as path_a_simulator).
        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        # Update current_turn_color from currentState.
        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        result = parse_step(event, acting_player)
        if result is None:
            continue

        for action_idx, forced_roll, forced_dev_card in result:
            if max_steps is not None and step_count >= max_steps:
                break

            if forced_roll is not None:
                last_roll = forced_roll

            try:
                stepper.step_and_override(action_idx, forced_roll, forced_dev_card)
            except Exception as exc:
                print(f"  step {step_count}: ERROR {exc}")
                step_count += 1
                continue

            step_count += 1
            action_desc = _action_label(action_idx, forced_roll, forced_dev_card)

            draw_board(
                stepper.env.unwrapped.game.state,
                ax,
                last_roll=last_roll,
                step=step_count,
                action_desc=action_desc,
            )
            fig.tight_layout(pad=0.3)
            frame = _fig_to_rgb(fig)
            writer.append_data(frame)

    writer.close()
    plt.close(fig)
    print(f"Wrote {step_count} frames → {output_mp4}")


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to an HxWx3 uint8 numpy array."""
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def _action_label(
    action_idx: int,
    forced_roll: Optional[int],
    forced_dev_card: Optional[str],
) -> str:
    """Return a short human-readable label for the action."""
    from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY

    try:
        action = ACTIONS_ARRAY[action_idx]
        action_type = str(action[0]).split(".")[-1]
        value = action[1]
        label = action_type if value is None else f"{action_type}({value})"
    except Exception:
        label = f"action[{action_idx}]"

    if forced_roll is not None:
        label += f" roll={forced_roll}"
    if forced_dev_card is not None:
        label += f" dev={forced_dev_card}"
    return label


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _json_path = sys.argv[1]
        _output_mp4 = sys.argv[2]
    else:
        _json_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "experiments",
            "rainbowzero",
            "catan",
            "replays",
            "ZxFhBPVr4KvuC3yN.json",
        )
        _output_mp4 = os.path.join(
            os.path.dirname(__file__),
            "..",
            "trajectory_verification.mp4",
        )

    print(f"Generating video from {_json_path} → {_output_mp4}")
    generate_video(_json_path, _output_mp4)
