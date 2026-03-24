"""Verifies Path A (Catanatron) and Path B (Direct Translator) against each other."""

import os
import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Ensure these imports point to your project modules
from custom_gym_envs.envs.catan import ACTIONS_ARRAY
from catanatron.models.enums import ActionType as AT
from catanatron.models.actions import generate_playable_actions
from json_parser import _texts_sorted, parse_step, reset_parser_state
from path_a_simulator import GodModeStepper
from path_b_translator import JsonStateTracker
import path_b_translator

DISCARD_IDX = ACTIONS_ARRAY.index((AT.DISCARD, None))

# Human-readable names for the 57 channels
CHANNEL_NAMES = [
    "P0_Settlement",
    "P0_City",
    "P0_Road",
    "P1_Settlement",
    "P1_City",
    "P1_Road",
    "Tile_WOOD",
    "Tile_BRICK",
    "Tile_SHEEP",
    "Tile_WHEAT",
    "Tile_ORE",
    "Tile_DESERT",
    "Dice_2",
    "Dice_3",
    "Dice_4",
    "Dice_5",
    "Dice_6",
    "Dice_8",
    "Dice_9",
    "Dice_10",
    "Dice_11",
    "Dice_12",
    "Robber",
    "Port_WOOD",
    "Port_BRICK",
    "Port_SHEEP",
    "Port_WHEAT",
    "Port_ORE",
    "Port_3:1",
    "Validity_Mask",
    "Roll_2",
    "Roll_3",
    "Roll_4",
    "Roll_5",
    "Roll_6",
    "Roll_7",
    "Roll_8",
    "Roll_9",
    "Roll_10",
    "Roll_11",
    "Roll_12",
    "Phase_Discard",
    "Phase_Robber",
    "P0_Rolled",
    "P1_Rolled",
    "Bank_WOOD",
    "Bank_BRICK",
    "Bank_SHEEP",
    "Bank_WHEAT",
    "Bank_ORE",
    "Bank_WOOD_Empty",
    "Bank_BRICK_Empty",
    "Bank_SHEEP_Empty",
    "Bank_WHEAT_Empty",
    "Bank_ORE_Empty",
    "P0_RoadDist",
    "P1_RoadDist",
]


def _board_config_from_initial_state(initial_state: dict) -> dict:
    board_config = {}
    tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
    for tid_str, tdata in tile_hex_states.items():
        board_config[int(tid_str)] = (tdata.get("type", 0), tdata.get("diceNumber", 0))
    return board_config


def _print_rich_desync_report(
    event_idx, event, action_idx, acting_player, stepper, obs_a, obs_b
):
    action_type, action_val = ACTIONS_ARRAY[action_idx]
    sc = event.get("stateChange", {})
    gls = sc.get("gameLogState", {})
    cs = sc.get("currentState", {})
    game = stepper.env.unwrapped.game

    print(f"\n{'='*85}")
    print(f"🚨 DESYNC DETECTED AT EVENT INDEX: {event_idx} 🚨")
    print(f"{'='*85}")

    print(f"\n[1] COLONIST ATTEMPTED ACTION")
    print(f"Action Attempted  : {action_type.name} {action_val} (Index: {action_idx})")
    print(f"Acting Player     : {acting_player}")
    print(f"Game Log Texts    : ")
    for k, v in _texts_sorted(gls):
        print(f"  - From {k}: {v}")

    print(f"\n[2] COLONIST INTERNAL STATE")
    print(f"Turn Color        : {cs.get('currentTurnPlayerColor', 'Unknown')}")
    print(f"Completed Turns   : {cs.get('completedTurns', 'Unknown')}")

    ps = sc.get("playerStates", {}).get(str(acting_player), {})
    rc = ps.get("resourceCards", {})
    if "cards" in rc and rc["cards"]:
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for c in rc["cards"]:
            counts[c] = counts.get(c, 0) + 1
        print(
            f"Hand Counts       : Wood:{counts.get(1,0)} Brick:{counts.get(2,0)} Sheep:{counts.get(3,0)} Wheat:{counts.get(4,0)} Ore:{counts.get(5,0)}"
        )
    else:
        print(
            f"Hand Counts       : Hidden or Empty (Total: {rc.get('count', rc.get('length', rc.get('total', 'Unknown')))})"
        )

    print(f"\n[3] CATANATRON INTERNAL STATE")
    print(f"Expected Agent    : {stepper.env.agent_selection}")
    print(f"Turn Color        : {game.state.current_color()}")
    print(f"Num Turns         : {getattr(game.state, 'num_turns', 'Unknown')}")
    stages = getattr(game.state, "turn_stages", [])
    print(
        f"Turn Stages       : {[s.name if hasattr(s, 'name') else str(s) for s in stages]}"
    )

    print(f"Catanatron Hand Counts (All Players):")
    expected_color = stepper.env.unwrapped.color_map[stepper.env.agent_selection]
    for cat_color in game.state.colors:
        p_idx = game.state.color_to_index[cat_color]
        p_str = f"P{p_idx}"
        w = game.state.player_state.get(f"{p_str}_WOOD_IN_HAND", 0)
        b = game.state.player_state.get(f"{p_str}_BRICK_IN_HAND", 0)
        s = game.state.player_state.get(f"{p_str}_SHEEP_IN_HAND", 0)
        wh = game.state.player_state.get(f"{p_str}_WHEAT_IN_HAND", 0)
        o = game.state.player_state.get(f"{p_str}_ORE_IN_HAND", 0)

        marker = " <--- (Acting Player)" if cat_color == expected_color else ""
        print(
            f"  - P{p_idx} (Color {cat_color.name}): Wood:{w} Brick:{b} Sheep:{s} Wheat:{wh} Ore:{o}{marker}"
        )

    print(f"\n[4] CATANATRON ALLOWED ACTIONS RIGHT NOW")
    valid_actions = stepper.env.unwrapped._get_valid_action_indices()
    if not valid_actions:
        print("  - NONE (Game might be over or completely stuck)")
    for va in valid_actions:
        v_type, v_val = ACTIONS_ARRAY[va]
        print(f"  - [{va}] {v_type.name} {v_val}")

    print(f"\n[5] TENSOR CHANNEL MISMATCH ANALYSIS")
    if obs_a is not None and obs_b is not None:
        mismatch_found = False
        max_channels = min(len(CHANNEL_NAMES), obs_a.shape[0], obs_b.shape[0])
        for ch in range(max_channels):
            if not np.allclose(obs_a[ch], obs_b[ch], atol=1e-5):
                mismatch_found = True
                print(f"  ❌ Channel {ch} ({CHANNEL_NAMES[ch]}) differs!")
                nz_a = list(zip(*np.where(obs_a[ch] != 0)))
                nz_b = list(zip(*np.where(obs_b[ch] != 0)))
                print(f"     Path A (Catanatron) active pixels : {nz_a[:5]}")
                print(f"     Path B (JSON Tracker) active pixels: {nz_b[:5]}")
        if not mismatch_found:
            print(
                "  ⚠️ Tensors matched perfectly in channels 0-44, but an action desync occurred."
            )
    else:
        print(
            "  ⚠️ No tensors provided to compare (Action desync, not a tensor mismatch)."
        )

    print(f"{'='*85}\n")


def process_and_verify_single(json_path: str) -> Optional[Dict[str, np.ndarray]]:
    reset_parser_state()
    with open(json_path, "r") as f:
        data = json.load(f).get("data", {})

    events = data.get("eventHistory", {}).get("events", [])
    play_order = data.get("playOrder", [])
    initial_state = data.get("eventHistory", {}).get("initialState", {})
    game_settings = data.get("gameSettings", initial_state.get("gameSettings", {}))

    n_players = len(play_order)
    stepper = GodModeStepper(num_players=n_players, representation="image")
    stepper.current_filename = Path(json_path).name

    discard_limit = int(game_settings.get("cardDiscardLimit", 7))
    tracker = JsonStateTracker(
        play_order,
        _board_config_from_initial_state(initial_state),
        discard_limit=discard_limit,
    )
    tracker.update(initial_state)

    tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
    if game_settings:
        stepper.sync_settings(game_settings)
    if tile_hex_states:
        stepper.set_board_layout(tile_hex_states, initial_state)
    map_state = initial_state.get("mapState", {})
    if map_state:
        stepper.sync_ports(map_state)

    # --- PORT FIX: Synchronize Path B's ports to match Path A's parsed Colonist ports ---
    tracker_port_nodes = {}
    from path_b_translator import _NODE_MAP

    for res, nids in stepper.env.unwrapped.game.state.board.map.port_nodes.items():
        tracker_port_nodes[res] = [_NODE_MAP[n] for n in nids if n in _NODE_MAP]
    path_b_translator._PORT_NODES = tracker_port_nodes
    # ------------------------------------------------------------------------------------

    current_turn_color = play_order[0]
    _colonist_to_color_idx = {c: i for i, c in enumerate(play_order)}

    obs_list, act_list = [], []
    done_parsing = False

    for ev_idx, event in enumerate(events):
        if done_parsing:
            break

        sc = event.get("stateChange", {})
        texts = _texts_sorted(sc.get("gameLogState", {}))

        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        if "playerStates" in sc:
            stepper.sync_resources(sc["playerStates"], play_order)

        env = stepper.env.unwrapped

        # ======================================================================
        # TODO: Update catanatron later to NOT have random discarding.
        # Catanatron currently handles discards automatically and randomly.
        # To align Colonist's asynchronous event order to Catanatron's strict
        # sequential order, we force Path B's phases to match Path A, and
        # execute discards here before evaluating the actual Colonist action.
        # ======================================================================
        while True:
            stepper._update_true_bank()
            stepper._recount_vps()
            env.game.playable_actions = generate_playable_actions(env.game.state)
            valid_actions = env._get_valid_action_indices()

            if len(valid_actions) == 1 and DISCARD_IDX in valid_actions:
                # Force Path B Phase to Discard
                tracker.is_discarding = True
                tracker.is_moving_robber = False
                obs_b_discard = tracker.to_tensor(acting_player)

                obs_list.append(obs_b_discard.copy())
                act_list.append(DISCARD_IDX)

                stepper.step_and_override(DISCARD_IDX, None, None)
                if stepper.env.unwrapped.game.winning_color() is not None:
                    break
            else:
                break

        # Sync Path B Phase to Catanatron's current True Phase
        stepper._update_true_bank()
        stepper._recount_vps()
        env.game.playable_actions = generate_playable_actions(env.game.state)
        valid_actions = env._get_valid_action_indices()

        tracker.is_discarding = DISCARD_IDX in valid_actions
        tracker.is_moving_robber = any(
            ACTIONS_ARRAY[va][0].name == "MOVE_ROBBER" for va in valid_actions
        )

        obs_b = tracker.to_tensor(acting_player)
        # ======================================================================

        c_idx = _colonist_to_color_idx[acting_player]
        catanatron_color = stepper.env.unwrapped.game.state.colors[c_idx]
        player_ports = stepper.env.unwrapped.game.state.board.get_player_port_resources(
            catanatron_color
        )
        current_ratios = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
        if None in player_ports:
            current_ratios = {k: 3 for k in current_ratios}
        _STR_TO_ENUM = {"WOOD": 1, "BRICK": 2, "SHEEP": 3, "WHEAT": 4, "ORE": 5}
        for port_res in player_ports:
            if port_res in _STR_TO_ENUM:
                current_ratios[_STR_TO_ENUM[port_res]] = 2

        result = parse_step(event, acting_player, player_ratios=current_ratios)
        tracker.update(sc)

        if "currentTurnPlayerColor" in sc.get("currentState", {}):
            current_turn_color = sc["currentState"]["currentTurnPlayerColor"]

        if result:
            # We explicitly filter out DISCARD actions parsed from the JSON because
            # we already forced them into the dataset via the Drain Loop above!
            filtered_result = [
                (idx, fr, fdc) for (idx, fr, fdc) in result if idx != DISCARD_IDX
            ]

            if not filtered_result:
                # If everything was filtered out, we still need to flush passive gains!
                stepper.flush_corrections()
            else:
                for i, (action_idx, forced_roll, forced_dev_card) in enumerate(
                    filtered_result
                ):
                    game = stepper.env.unwrapped.game
                    stepper._update_true_bank()
                    stepper._recount_vps()
                    game.playable_actions = generate_playable_actions(game.state)

                    valid_actions = stepper.env.unwrapped._get_valid_action_indices()
                    action_type, _ = ACTIONS_ARRAY[action_idx]

                    if action_type.name == "END_TURN" and action_idx not in valid_actions:
                        continue

                    if action_idx not in valid_actions:
                        print(f"\n❌ ACTION DESYNC IN: {Path(json_path).name}")
                        _print_rich_desync_report(
                            ev_idx, event, action_idx, acting_player, stepper, None, None
                        )
                        return None

                    agent_id = stepper.env.unwrapped.agent_map[catanatron_color]
                    obs_a = stepper.env.unwrapped.observe(agent_id)["observation"]

                    # Compare Channels 0-44 (Exclude bank and road lengths which are computed differently)
                    if not np.allclose(obs_a[:45], obs_b[:45], atol=1e-5):
                        print(f"\n❌ TENSOR MISMATCH IN: {Path(json_path).name}")
                        _print_rich_desync_report(
                            ev_idx,
                            event,
                            action_idx,
                            acting_player,
                            stepper,
                            obs_a,
                            obs_b,
                        )
                        return None

                    obs_list.append(obs_b.copy())
                    act_list.append(action_idx)

                    if stepper.env.unwrapped.game.winning_color() is not None:
                        done_parsing = True
                        break

                    stepper.step_and_override(action_idx, forced_roll, forced_dev_card)
                    if i == len(filtered_result) - 1:
                        stepper.flush_corrections()
        else:
            # FIX: Even if there were no actions parsed (e.g. player trades, passive
            # resource gains from other players rolling), flush the pending resources!
            stepper.flush_corrections()

    if not obs_list:
        return None
    return {"observations": np.stack(obs_list), "actions": np.array(act_list)}


def generate_verified_dataset(replays_dir: str, output_path: str):
    """Processes an entire directory of replays and saves the verified dataset."""
    json_files = list(Path(replays_dir).glob("*.json"))
    print(f"Found {len(json_files)} replays. Starting generation & verification...")

    verified_obs, verified_acts = [], []
    failed = 0

    for idx, fpath in enumerate(json_files, 1):
        try:
            dataset = process_and_verify_single(str(fpath))
            if dataset is not None:
                verified_obs.append(dataset["observations"])
                verified_acts.append(dataset["actions"])
                print(
                    f"[{idx}/{len(json_files)}] ✅ Verified & Processed: {fpath.name}"
                )
            else:
                failed += 1
        except Exception as e:
            print(f"[{idx}/{len(json_files)}] ⚠️ Error on {fpath.name}: {str(e)}")
            failed += 1

    if verified_obs:
        final_dataset = {
            "observations": np.concatenate(verified_obs, axis=0),
            "actions": np.concatenate(verified_acts, axis=0),
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with gzip.open(output_path, "wb") as f:
            pickle.dump(final_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"\n🎉 Success! Saved {len(final_dataset['actions'])} verified steps to {output_path}"
        )
    else:
        print("\n⚠️ No replays were successfully verified.")


if __name__ == "__main__":
    # UPDATE THESE PATHS TO MATCH YOUR SYSTEM
    REPLAYS_DIR = "/Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/experiments/catan/offline_data/replays"

    # Let's test on just ONE file first to see the exact channel breakdown
    test_file = os.path.join(REPLAYS_DIR, "FBb37KYyzd52bpjx.json")

    if os.path.exists(test_file):
        print(f"Running highly detailed diagnostic on {test_file}...")
        process_and_verify_single(test_file)
    else:
        print("Test file not found! Check your directory path.")
