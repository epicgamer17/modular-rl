"""Directly converts Colonist.io replays to Catanatron (State, Action) pairs."""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# Import your existing robust mappers
from path_b_translator import JsonStateTracker
from json_parser import parse_step, reset_parser_state, _texts_sorted


def _board_config_from_initial_state(initial_state: dict) -> dict:
    """Extract full board layout mapping."""
    board_config = {}
    tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
    for tid_str, tdata in tile_hex_states.items():
        board_config[int(tid_str)] = (tdata.get("type", 0), tdata.get("diceNumber", 0))
    return board_config


def convert_replay(json_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Converts a single Colonist JSON replay into Catanatron tensors and actions."""

    with open(json_path, "r") as f:
        data = json.load(f).get("data", {})

    events = data.get("eventHistory", {}).get("events", [])
    play_order = data.get("playOrder", [])
    initial_state = data.get("eventHistory", {}).get("initialState", {})

    # Reset tracking singletons
    reset_parser_state()

    # Initialize the Path B State Tracker (Builds Catanatron Tensors directly from JSON)
    board_config = _board_config_from_initial_state(initial_state)
    tracker = JsonStateTracker(play_order, board_config=board_config)
    tracker.update(initial_state)

    # Determine Winner
    end_game = data.get("eventHistory", {}).get("endGameState", {})
    winner_color = None
    for color_str, pdata in end_game.get("players", {}).items():
        if pdata.get("winningPlayer"):
            winner_color = int(color_str)
            break

    current_turn_color = play_order[0]

    obs_list = []
    act_list = []
    rewards_list = []

    for event in events:
        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        # 1. Determine who is acting in this event
        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        # 2. Get the Catanatron State Tensor BEFORE the action alters the board
        # Shape: (57, 22, 14) for 2 players
        state_tensor = tracker.to_tensor(acting_player)

        # 3. Parse the Colonist event into Catanatron Action Indices
        catanatron_actions = parse_step(event, acting_player)

        # 4. Update the tracker with the consequences of the action for the NEXT step
        tracker.update(sc)

        # 5. Advance turn if required
        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        # 6. Save the (State, Action) mapping
        if catanatron_actions is not None:
            # parse_step can return multiple atomic actions for bundled colonist events (like trades)
            for action_idx, _, _ in catanatron_actions:
                obs_list.append(state_tensor.copy())
                act_list.append(action_idx)

                # Assign reward (1.0 for winner, -1.0 for losers)
                reward = 1.0 if acting_player == winner_color else -1.0
                rewards_list.append(reward)

    if not obs_list:
        return None

    # Stack into final datasets expected by your PyTorch pipeline
    return {
        "observations": np.stack(obs_list, axis=0).astype(np.float32),
        "actions": np.array(act_list, dtype=np.int64),
        "rewards": np.array(rewards_list, dtype=np.float32),
    }


def process_directory(replays_dir: str, output_file: str):
    """Processes all JSONs and saves a combined dataset."""
    all_obs, all_acts, all_rewards = [], [], []
    json_files = list(Path(replays_dir).glob("*.json"))

    print(f"Found {len(json_files)} replays. Starting conversion...")

    success_count = 0
    for idx, fpath in enumerate(json_files, 1):
        try:
            dataset = convert_replay(str(fpath))
            if dataset is not None:
                all_obs.append(dataset["observations"])
                all_acts.append(dataset["actions"])
                all_rewards.append(dataset["rewards"])
                success_count += 1
            print(f"[{idx}/{len(json_files)}] Processed: {fpath.name}")
        except Exception as e:
            print(f"[{idx}/{len(json_files)}] Failed: {fpath.name} - {str(e)}")

    if not all_obs:
        print("No valid data extracted.")
        return

    # Combine all games into master arrays
    final_dataset = {
        "observations": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_acts, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
    }

    # Save to disk (for PyTorch DataLoader)
    import pickle, gzip

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with gzip.open(output_file, "wb") as f:
        pickle.dump(final_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSuccessfully compiled {success_count} games.")
    print(f"Total State-Action pairs: {len(final_dataset['actions'])}")
    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    REPLAYS_DIR = "offline_data/replays"
    OUTPUT_FILE = "offline_data/catanatron_converted_dataset.pkl.gz"
    process_directory(REPLAYS_DIR, OUTPUT_FILE)
