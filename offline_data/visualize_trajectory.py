"""Visualize a colonist.io game replay as an MP4 video."""

import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(
    0, os.path.dirname(__file__)
)  # bare imports: json_parser, path_a_simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0, "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron"
)

import imageio.v2 as imageio

from json_parser import _texts_sorted, parse_step
from path_a_simulator import GodModeStepper


def _ensure_hwc(frame: np.ndarray) -> np.ndarray:
    """Transpose (W, H, 3) → (H, W, 3) if pygame surfarray layout is detected."""
    if frame.ndim == 3 and frame.shape[0] > frame.shape[1]:
        return np.transpose(frame, (1, 0, 2))
    return frame


def generate_video(
    json_path: str,
    output_mp4: str,
    fps: int = 3,
    max_steps: Optional[int] = None,
) -> None:
    with open(json_path) as fh:
        data = json.load(fh)

    d = data["data"]
    events = d["eventHistory"]["events"]
    play_order: list[int] = d["playOrder"]

    stepper = GodModeStepper(render_mode="rgb_array")
    current_turn_color: int = play_order[0]

    writer = imageio.get_writer(output_mp4, fps=fps, macro_block_size=None)

    # Sync board layout and ports. Look in both possible JSON locations.
    # Sync board layout and ports. Look in both possible JSON locations.
    initial_state = d.get(
        "initialState", d.get("eventHistory", {}).get("initialState", {})
    )
    map_state = initial_state.get("mapState", {})
    tile_hex_states = map_state.get("tileHexStates", {})

    if tile_hex_states:
        # Pass initial_state so we can extract the true robber starting index
        stepper.set_board_layout(tile_hex_states, initial_state)
    else:
        print(
            "WARNING: Could not find tileHexStates in JSON! The board will not be synced."
        )

    if map_state:
        stepper.sync_ports(map_state)

    step_count = 0

    # Render the initial board as frame 0.
    frame = stepper.env.render()
    if isinstance(frame, np.ndarray):
        writer.append_data(_ensure_hwc(frame))

    for ev_idx, event in enumerate(events):
        if max_steps is not None and step_count >= max_steps:
            break

        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        if "playerStates" in sc:
            stepper.sync_resources(sc["playerStates"], play_order)

        result = parse_step(event, acting_player)
        if result is None:
            continue

        for action_idx, forced_roll, forced_dev_card in result:
            if max_steps is not None and step_count >= max_steps:
                break

            try:
                stepper.step_and_override(action_idx, forced_roll, forced_dev_card)
            except Exception as exc:
                print(f"  step {step_count}: ERROR {exc}")
                step_count += 1
                continue

            agent = stepper.env.agent_selection
            if stepper.env.terminations[agent] or stepper.env.truncations[agent]:
                print(f"  step {step_count}: Terminated/Truncated for agent {agent}")
                break

            step_count += 1

            frame = stepper.env.render()
            if isinstance(frame, np.ndarray):
                writer.append_data(_ensure_hwc(frame))

    writer.close()
    print(f"Wrote {step_count} frames → {output_mp4}")


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
