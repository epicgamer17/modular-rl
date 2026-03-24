"""Generate the offline Catan imitation-learning dataset."""

from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0,
    "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron",
)

from json_parser import _texts_sorted, parse_step, reset_parser_state
from path_a_simulator import GodModeStepper
from path_b_translator import JsonStateTracker
from visualize_trajectory import generate_video
from catanatron.models.actions import generate_playable_actions

from catanatron.models.enums import ActionType as AT
from custom_gym_envs.envs.catan import ACTIONS_ARRAY

# The globally recognized action space index for DISCARD
DISCARD_IDX = ACTIONS_ARRAY.index((AT.DISCARD, None))

# ---------------------------------------------------------------------------
# Channel names (57 for n=2)
# ---------------------------------------------------------------------------

_CHANNEL_NAMES: list[str] = [
    "P0 (ME) Settlements",
    "P0 (ME) Cities",
    "P0 (ME) Roads",
    "P1 (NEXT) Settlements",
    "P1 (NEXT) Cities",
    "P1 (NEXT) Roads",
    "Tile Resource: WOOD",
    "Tile Resource: BRICK",
    "Tile Resource: SHEEP",
    "Tile Resource: WHEAT",
    "Tile Resource: ORE",
    "Tile Resource: DESERT",
    "Tile Dice: 2",
    "Tile Dice: 3",
    "Tile Dice: 4",
    "Tile Dice: 5",
    "Tile Dice: 6",
    "Tile Dice: 8",
    "Tile Dice: 9",
    "Tile Dice: 10",
    "Tile Dice: 11",
    "Tile Dice: 12",
    "Robber",
    "Port: WOOD (2:1)",
    "Port: BRICK (2:1)",
    "Port: SHEEP (2:1)",
    "Port: WHEAT (2:1)",
    "Port: ORE (2:1)",
    "Port: 3:1",
    "Validity Mask",
    "Last Roll: 2",
    "Last Roll: 3",
    "Last Roll: 4",
    "Last Roll: 5",
    "Last Roll: 6",
    "Last Roll: 7",
    "Last Roll: 8",
    "Last Roll: 9",
    "Last Roll: 10",
    "Last Roll: 11",
    "Last Roll: 12",
    "Game Phase: IS_DISCARDING",
    "Game Phase: IS_MOVING_ROBBER",
    "Game Phase: P0 (ME) HAS_ROLLED",
    "Game Phase: P1 (NEXT) HAS_ROLLED",
    "Bank Normalised: WOOD",
    "Bank Normalised: BRICK",
    "Bank Normalised: SHEEP",
    "Bank Normalised: WHEAT",
    "Bank Normalised: ORE",
    "Bank Empty: WOOD",
    "Bank Empty: BRICK",
    "Bank Empty: SHEEP",
    "Bank Empty: WHEAT",
    "Bank Empty: ORE",
    "Road Distance: P0 (ME)",
    "Road Distance: P1 (NEXT)",
]


def _channel_name(ch: int) -> str:
    if ch < len(_CHANNEL_NAMES):
        return _CHANNEL_NAMES[ch]
    return f"Unknown channel {ch}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _board_config_from_initial_state(initial_state: dict) -> dict[int, tuple[int, int]]:
    """Extract full board layout from eventHistory.initialState.mapState.tileHexStates."""
    board_config: dict[int, tuple[int, int]] = {}
    tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
    for tid_str, tdata in tile_hex_states.items():
        web_tile_id = int(tid_str)
        res_enum = tdata.get("type", 0)
        dice_num = tdata.get("diceNumber", 0)
        board_config[web_tile_id] = (res_enum, dice_num)
    return board_config


def _extract_acting_player(
    texts: list[tuple[Optional[int], dict]], current_turn_color: int
) -> int:
    for _, t in texts:
        pc = t.get("playerColor")
        if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
            return pc
    return current_turn_color


def _find_winner(events: list[dict]) -> Optional[int]:
    for ev in events:
        sc = ev.get("stateChange", {})
        for _, v in sc.get("gameLogState", {}).items():
            t = v.get("text", {})
            if isinstance(t, dict) and t.get("type") == 45:
                return t.get("playerColor")
    return None


# ---------------------------------------------------------------------------
# Path A + B reconciliation helpers
# ---------------------------------------------------------------------------


def _report_mismatch(obs_a: np.ndarray, obs_b: np.ndarray, step: int) -> None:
    """Print a detailed per-channel breakdown of the first differing dynamic channel."""
    print(f"\n{'='*70}")
    print(f"MISMATCH at step {step}")
    print(f"{'='*70}")
    mismatch_found = False
    static_set = set(_BOARD_DEPENDENT_CHANNELS)
    for ch in range(obs_a.shape[0]):
        if ch in static_set:
            continue  # skip static board layout channels
        plane_a = obs_a[ch]
        plane_b = obs_b[ch]
        if not np.allclose(plane_a, plane_b, atol=1e-5):
            name = _channel_name(ch)
            max_abs_diff = np.abs(plane_a - plane_b).max()
            n_pixels = int((plane_a != plane_b).sum())
            print(f"\nChannel {ch}: {name}")
            print(f"  max |a-b|={max_abs_diff:.6f}  differing pixels={n_pixels}")
            nz_a = list(zip(*np.where(plane_a != 0)))
            nz_b = list(zip(*np.where(plane_b != 0)))
            if nz_a or nz_b:
                print(f"  Path A non-zero at: {nz_a[:10]}")
                print(f"  Path B non-zero at: {nz_b[:10]}")
            else:
                print("  Both planes are all-zeros but differ by float precision")
            mismatch_found = True
            break

    if not mismatch_found:
        print("Mismatch is sub-threshold per-channel but allclose failed overall.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# verify_trajectory
# ---------------------------------------------------------------------------

_BOARD_DEPENDENT_CHANNELS = (
    list(range(6, 29))  # 6–28: tiles, robber, ports
    + list(range(30, 41))  # 30–40: last roll
    + list(range(45, 55))  # 45–54: bank state
)


def verify_trajectory(json_path: str, max_steps: int = 9999) -> None:
    with open(json_path) as f:
        data_root = json.load(f)
        data = data_root["data"]

    events: list[dict] = data["eventHistory"]["events"]
    play_order: list[int] = data["playOrder"]
    initial_state: dict = data["eventHistory"].get("initialState", {})

    stepper = GodModeStepper(representation="mixed")
    board_config = _board_config_from_initial_state(initial_state)
    tracker = JsonStateTracker(play_order, board_config=board_config)
    tracker.update(initial_state)

    game_settings = data.get("gameSettings", initial_state.get("gameSettings", {}))
    if game_settings:
        stepper.sync_settings(game_settings)

    _color_idx_to_colonist: dict[int, int] = {i: c for i, c in enumerate(play_order)}
    current_turn_color: int = play_order[0]
    step = 0

    n_channels = 57
    dynamic_mask = np.ones(n_channels, dtype=bool)
    for ch in _BOARD_DEPENDENT_CHANNELS:
        dynamic_mask[ch] = False
    dynamic_ch_indices = np.where(dynamic_mask)[0]

    print(f"Verifying: {Path(json_path).name}  ({len(events)} events)")
    print(f"Board tiles from initial state: {len(board_config)}/19")
    print(
        f"Comparing {dynamic_mask.sum()} channels: buildings(0–5), mask(29), phase(41–44), road(55–56)"
    )
    print(f"\n{'step':<5} {'ev':<6} {'actor':<8} {'action_idx':<12} status")
    print("-" * 55)

    for ev_idx, event in enumerate(events):
        if step >= max_steps:
            break

        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        acting_player = _extract_acting_player(texts, current_turn_color)

        obs_b_pre = tracker.to_tensor(acting_player)

        if "playerStates" in sc:
            stepper.sync_resources(sc["playerStates"], play_order)

        # Apply auto-drain fix here as well to keep the verifier from crashing on 7s
        while True:
            env = stepper.env.unwrapped
            stepper._update_true_bank()
            stepper._recount_vps()
            env.game.playable_actions = generate_playable_actions(env.game.state)
            valid_actions = env._get_valid_action_indices()

            if len(valid_actions) == 1 and DISCARD_IDX in valid_actions:
                if stepper.env.unwrapped.game.winning_color() is not None:
                    break
                stepper.step_and_override(DISCARD_IDX, None, None)
            else:
                break

        result = parse_step(event, acting_player)
        tracker.update(sc)

        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        if result is not None:
            if not tracker._initial_phase:
                remaining = len(events) - ev_idx - 1
                print(
                    f"\nInitial placement phase complete at ev{ev_idx}. "
                    f"Stopping comparison ({remaining} events remaining).\n"
                    f"Post-placement comparison is unsound: catanatron uses a random\n"
                    f"board so paid actions become invalid after resource production diverges."
                )
                break

            filtered_result = [r for r in result if r[0] != DISCARD_IDX]
            for action_idx, forced_roll, forced_dev_card in filtered_result:
                if stepper.env.unwrapped.game.winning_color() is not None:
                    break

                obs_a_full = stepper.step_and_override(
                    action_idx, forced_roll, forced_dev_card
                )
                obs_a = obs_a_full["board"]

                env_agent = stepper.env.agent_selection
                env_agent_idx = int(env_agent.split("_")[-1])
                env_next_colonist = _color_idx_to_colonist[env_agent_idx]
                obs_b = tracker.to_tensor(env_next_colonist)

                try:
                    np.testing.assert_allclose(
                        obs_a[dynamic_ch_indices],
                        obs_b[dynamic_ch_indices],
                        atol=1e-5,
                        rtol=0,
                    )
                    print(
                        f"{step:<5} ev{ev_idx:<4} p{acting_player:<7} {action_idx:<12} OK"
                    )
                except AssertionError:
                    print(
                        f"{step:<5} ev{ev_idx:<4} p{acting_player:<7} {action_idx:<12} FAIL"
                    )
                    _report_mismatch(obs_a, obs_b, step)
                    raise

                step += 1
                if step >= max_steps:
                    break

        stepper.flush_corrections()

    print(
        f"\nVerification complete: {step} initial-placement steps checked — ALL MATCH"
    )


# ---------------------------------------------------------------------------
# process_all_replays
# ---------------------------------------------------------------------------


def _load_replay_json(source) -> Optional[dict]:
    try:
        if isinstance(source, (str, Path)):
            with open(source) as f:
                return json.load(f)
        else:
            with source as f:
                return json.loads(f.read().decode("utf-8"))
    except Exception as exc:
        print(f"  [WARN] Failed to load replay: {exc}")
        return None


def _process_single_replay(raw: dict, filename: str = "Unknown") -> Optional[dict]:
    """Uses GodModeStepper (Path A) to generate observations."""
    reset_parser_state()
    data = raw.get("data", raw)
    events = data["eventHistory"]["events"]
    play_order = data["playOrder"]
    initial_state = data["eventHistory"].get("initialState", {})

    # 1. EXTRACT PLAYER COUNT
    game_settings = data.get("gameSettings", initial_state.get("gameSettings", {}))
    n_players = game_settings.get("maxPlayers", len(play_order))

    # 2. INITIALIZE STEPPER
    # Ensure your env supports 'num_players' argument
    stepper = GodModeStepper(num_players=n_players, representation="image")
    stepper.current_filename = filename
    winner = _find_winner(events)

    tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
    if game_settings:
        stepper.sync_settings(game_settings)
    if tile_hex_states:
        stepper.set_board_layout(tile_hex_states, initial_state)
    map_state = initial_state.get("mapState", {})
    if map_state:
        stepper.sync_ports(map_state)

    current_turn_color: int = play_order[0]
    _colonist_to_color_idx = {c: i for i, c in enumerate(play_order)}

    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    act_list: list[int] = []
    actor_list: list[int] = []

    done_parsing = False

    for event in events:
        if done_parsing:
            break

        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        acting_player = _extract_acting_player(texts, current_turn_color)

        if "playerStates" in sc:
            stepper.sync_resources(sc["playerStates"], play_order)

        # --- DRAIN PENDING DISCARDS ---
        while True:
            env = stepper.env.unwrapped
            stepper._update_true_bank()
            stepper._recount_vps()
            env.game.playable_actions = generate_playable_actions(env.game.state)
            valid_actions = env._get_valid_action_indices()

            if len(valid_actions) == 1 and DISCARD_IDX in valid_actions:
                current_agent = env.agent_selection
                aec_obs = env.observe(current_agent)

                obs_list.append(aec_obs["observation"])
                mask_list.append(aec_obs["action_mask"])
                act_list.append(DISCARD_IDX)

                c_color = env.color_map[current_agent]
                colonist_actor = play_order[env.game.state.colors.index(c_color)]
                actor_list.append(colonist_actor)

                if stepper.env.unwrapped.game.winning_color() is not None:
                    break
                stepper.step_and_override(DISCARD_IDX, None, None)
            else:
                break

        # --- NEW: EXTRACT TRADE RATIOS FOR THE PARSER ---
        # We get the acting player's specific ratios from the current Catanatron state
        c_idx = _colonist_to_color_idx[acting_player]
        game_state = stepper.env.unwrapped.game.state

        # Mapping Catanatron's internal string-based ratios back to Colonist's 1-5 enums
        current_ratios = {
            1: game_state.player_state[f"P{c_idx}_WOOD_TRADE_RATIO"],
            2: game_state.player_state[f"P{c_idx}_BRICK_TRADE_RATIO"],
            3: game_state.player_state[f"P{c_idx}_SHEEP_TRADE_RATIO"],
            4: game_state.player_state[f"P{c_idx}_WHEAT_TRADE_RATIO"],
            5: game_state.player_state[f"P{c_idx}_ORE_TRADE_RATIO"],
        }

        # Pass ratios to the parser to handle combined/multiple trades correctly
        result = parse_step(event, acting_player, player_ratios=current_ratios)

        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        if result is not None:
            filtered_result = [
                (idx, fr, fdc) for (idx, fr, fdc) in result if idx != DISCARD_IDX
            ]

            for i, (action_idx, forced_roll, forced_dev_card) in enumerate(
                filtered_result
            ):
                game = stepper.env.unwrapped.game
                stepper._update_true_bank()
                stepper._recount_vps()
                game.playable_actions = generate_playable_actions(game.state)

                c_idx = _colonist_to_color_idx[acting_player]
                catanatron_color = game.state.colors[c_idx]

                agent_id = stepper.env.unwrapped.agent_map[catanatron_color]
                aec_obs = stepper.env.unwrapped.observe(agent_id)

                obs_list.append(aec_obs["observation"])
                mask_list.append(aec_obs["action_mask"])
                act_list.append(action_idx)
                actor_list.append(acting_player)

                if stepper.env.unwrapped.game.winning_color() is not None:
                    done_parsing = True
                    break

                stepper.step_and_override(action_idx, forced_roll, forced_dev_card)

                if i == len(filtered_result) - 1:
                    stepper.flush_corrections()

                if stepper.env.unwrapped.game.winning_color() is not None:
                    done_parsing = True
                    break

            if not filtered_result:
                stepper.flush_corrections()
                if stepper.env.unwrapped.game.winning_color() is not None:
                    done_parsing = True
        else:
            stepper.flush_corrections()
            if stepper.env.unwrapped.game.winning_color() is not None:
                done_parsing = True

    stepper.flush_corrections()

    if not obs_list:
        return None

    T = len(obs_list)
    observations = np.stack(obs_list, axis=0).astype(np.float32)
    masks = np.stack(mask_list, axis=0).astype(np.int8)
    actions = np.array(act_list, dtype=np.int32)
    rewards = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=bool)

    dones[-1] = True
    if winner is not None:
        for i, actor in enumerate(actor_list):
            if dones[i]:
                rewards[i] = 1.0 if actor == winner else -1.0

        if actor_list[-1] != winner:
            for i in range(T - 1, -1, -1):
                if actor_list[i] == winner:
                    rewards[i] = 1.0
                    dones[i] = True
                    rewards[T - 1] = -1.0
                    break

    return {
        "observations": observations,
        "action_masks": masks,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def process_all_replays(replays_dir: str, output_path: str) -> None:
    episodes: list[dict] = []
    failed = 0

    zip_path: Optional[str] = None
    inner_prefix: str = ""

    path_obj = Path(replays_dir)

    parts = Path(replays_dir).parts
    for i, part in enumerate(parts):
        candidate = Path(*parts[: i + 1])
        if str(candidate).endswith(".zip") and candidate.is_file():
            zip_path = str(candidate)
            inner_prefix = str(Path(*parts[i + 1 :])) if i + 1 < len(parts) else ""
            break

    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [
                n
                for n in zf.namelist()
                if n.endswith(".json")
                and (inner_prefix == "" or n.startswith(inner_prefix))
            ]
            total = len(names)
            print(f"Processing {total} replays from {zip_path} ...")
            for idx, name in enumerate(names, 1):
                if idx % 20 == 0 or idx == total:
                    print(f"  {idx}/{total} ...", end="\r")
                raw_bytes = zf.read(name)
                try:
                    raw = json.loads(raw_bytes.decode("utf-8"))
                except Exception:
                    failed += 1
                    continue
                ep = _process_single_replay(raw, filename=fpath.name)
                if ep is not None:
                    episodes.append(ep)
                else:
                    failed += 1
    else:
        json_files = sorted(path_obj.glob("*.json"))
        total = len(json_files)
        print(f"Processing {total} replays from {replays_dir} ...")
        for idx, fpath in enumerate(json_files, 1):
            raw = _load_replay_json(fpath)
            if raw is None:
                failed += 1
                continue

            try:
                # Pass the filename to the processor
                ep = _process_single_replay(raw, filename=fpath.name)
                if ep is not None:
                    episodes.append(ep)
                else:
                    failed += 1
            except ValueError as e:
                print(f"\n[DEBUG] Automatically visualizing desync for: {fpath.name}")
                video_output = os.path.join(
                    os.path.dirname(output_path), f"debug_{fpath.stem}.mp4"
                )

                # Call your visualization script logic
                try:
                    generate_video(str(fpath), video_output)
                    print(f"[DEBUG] Video saved to: {video_output}")
                except Exception as viz_err:
                    print(f"[DEBUG] Failed to generate debug video: {viz_err}")

                # Stop execution to allow you to inspect the video
                raise e
    print(f"\nDone. {len(episodes)} episodes saved, {failed} skipped.")

    if not episodes:
        print("No episodes successfully processed.")
        return

    total_steps = sum(len(ep["actions"]) for ep in episodes)
    print(f"Total steps: {total_steps}")
    print(f"Avg steps per episode: {total_steps / max(len(episodes), 1):.1f}")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".gz"):
        with gzip.open(output_path, "wb") as f:
            pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(output_path, "wb") as f:
            pickle.dump(episodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path_obj.stat().st_size / 1e6
    print(f"Saved to {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    REPLAYS_DIR = os.path.join(
        os.path.dirname(__file__),
        "..",
        "experiments",
        "rainbowzero",
        "catan",
        "replays",
    )

    TEST_REPLAY = os.path.join(
        REPLAYS_DIR,
        "ZxFhBPVr4KvuC3yN.json",
    )

    OUTPUT_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "experiments",
        "rainbowzero",
        "catan",
        "catan_human_data.pkl.gz",
    )

    print("=" * 60)
    print("STEP 1: Verifying Path A ↔ Path B on test replay ...")
    print("=" * 60)
    try:
        verify_trajectory(TEST_REPLAY)
    except AssertionError as exc:
        print(f"\nVerification FAILED: {exc}")
        raise SystemExit(1)

    print("\nVerification Successful\n")

    print("=" * 60)
    print("STEP 2: Generating full dataset ...")
    print("=" * 60)
    process_all_replays(REPLAYS_DIR, OUTPUT_PATH)
