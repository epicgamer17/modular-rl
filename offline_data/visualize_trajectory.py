"""Visualize a colonist.io game replay as an MP4 video."""

import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0, "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron"
)

import imageio.v2 as imageio

# IMPORT THE NEW RESET FUNCTION
from json_parser import _texts_sorted, parse_step, reset_parser_state
from path_a_simulator import GodModeStepper


def _ensure_hwc(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] > frame.shape[1]:
        return np.transpose(frame, (1, 0, 2))
    return frame


def generate_video(
    json_path: str,
    output_mp4: str,
    fps: int = 3,
    max_steps: Optional[int] = None,
) -> None:

    # CLEAR THE PARSER CACHE FOR THE NEW GAME
    reset_parser_state()

    with open(json_path) as fh:
        data = json.load(fh)

    d = data["data"]
    events = d["eventHistory"]["events"]
    play_order: list[int] = d["playOrder"]

    stepper = GodModeStepper(render_mode="rgb_array")
    current_turn_color: int = play_order[0]

    writer = imageio.get_writer(output_mp4, fps=fps, macro_block_size=None)

    initial_state = d.get(
        "initialState", d.get("eventHistory", {}).get("initialState", {})
    )
    map_state = initial_state.get("mapState", {})
    tile_hex_states = map_state.get("tileHexStates", {})

    game_settings = d.get("gameSettings", initial_state.get("gameSettings", {}))
    if game_settings:
        stepper.sync_settings(game_settings)

    if tile_hex_states:
        stepper.set_board_layout(tile_hex_states, initial_state)

    if map_state:
        stepper.sync_ports(map_state)

    step_count = 0
    done = False

    frame = stepper.env.render()
    if isinstance(frame, np.ndarray):
        writer.append_data(_ensure_hwc(frame))

    for ev_idx, event in enumerate(events):
        if done:
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

        if result is not None:
            for i, (action_idx, forced_roll, forced_dev_card) in enumerate(result):
                if max_steps is not None and step_count >= max_steps:
                    done = True
                    break

                stepper.step_and_override(action_idx, forced_roll, forced_dev_card)

                # FLUSH IMMEDIATELY IF LAST ACTION IN BATCH BEFORE RENDERING
                if i == len(result) - 1:
                    stepper.flush_corrections()

                step_count += 1

                # Render AFTER the flush so the video captures the perfectly synced state
                frame = stepper.env.render()
                if isinstance(frame, np.ndarray):
                    writer.append_data(_ensure_hwc(frame))
        else:
            # Even if there were no actions, flush passive resource gains!
            stepper.flush_corrections()

    # --- ADD THIS BLOCK AFTER THE LOOP ---
    # Flush any final terminal state corrections (like the game-over VP reveal)
    stepper.flush_corrections()

    # Render the absolute final terminal state
    final_frame = stepper.env.render()
    if isinstance(final_frame, np.ndarray):
        hwc_final = _ensure_hwc(final_frame)
        # Duplicate the final frame to "pause" the video at the end for 3 seconds
        pause_frames = fps * 3
        for _ in range(pause_frames):
            writer.append_data(hwc_final)
    # -------------------------------------

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
    generate_video(_json_path, _output_mp4)
