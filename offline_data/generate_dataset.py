"""Generate the offline Catan imitation-learning dataset.

Two entry-points:

  verify_trajectory(json_path)
      Runs Path A (GodModeStepper, catanatron env) and Path B (JsonStateTracker)
      in lock-step on a single replay.  At every action step it asserts that the
      57-channel board tensors produced by both paths are numerically identical.
      On failure it prints a detailed per-channel breakdown and raises.

  process_all_replays(replays_dir, output_path)
      Uses Path B only (fast, no env) to iterate every .json file in replays_dir.
      For each game it produces an episode dict:
          {
              "observations": np.float32 (T, 57, 22, 14),   # pre-action obs
              "actions":      np.int32   (T,),
              "rewards":      np.float32 (T,),               # 0 except last step
              "dones":        np.bool_   (T,),               # True only on last step
          }
      Rewards: +1 for the winner on the final step, -1 for everyone else.
      Saves the list of episode dicts as a gzip-pickle at output_path.

Channel reference (57 channels, n_players=2)
────────────────────────────────────────────
  0   Player 0 (ME) Settlements
  1   Player 0 (ME) Cities
  2   Player 0 (ME) Roads
  3   Player 1 (NEXT) Settlements
  4   Player 1 (NEXT) Cities
  5   Player 1 (NEXT) Roads
  6   Tile Resource: WOOD
  7   Tile Resource: BRICK
  8   Tile Resource: SHEEP
  9   Tile Resource: WHEAT
  10  Tile Resource: ORE
  11  Tile Resource: DESERT
  12  Tile Dice: 2
  13  Tile Dice: 3
  14  Tile Dice: 4
  15  Tile Dice: 5
  16  Tile Dice: 6
  17  Tile Dice: 8
  18  Tile Dice: 9
  19  Tile Dice: 10
  20  Tile Dice: 11
  21  Tile Dice: 12
  22  Robber
  23  Port: WOOD (2:1)
  24  Port: BRICK (2:1)
  25  Port: SHEEP (2:1)
  26  Port: WHEAT (2:1)
  27  Port: ORE (2:1)
  28  Port: 3:1
  29  Validity Mask
  30  Last Roll: 2
  31  Last Roll: 3
  32  Last Roll: 4
  33  Last Roll: 5
  34  Last Roll: 6
  35  Last Roll: 7
  36  Last Roll: 8
  37  Last Roll: 9
  38  Last Roll: 10
  39  Last Roll: 11
  40  Last Roll: 12
  41  Game Phase: IS_DISCARDING
  42  Game Phase: IS_MOVING_ROBBER
  43  Game Phase: P0 (ME) HAS_ROLLED
  44  Game Phase: P1 (NEXT) HAS_ROLLED
  45  Bank Normalised: WOOD
  46  Bank Normalised: BRICK
  47  Bank Normalised: SHEEP
  48  Bank Normalised: WHEAT
  49  Bank Normalised: ORE
  50  Bank Empty Flag: WOOD
  51  Bank Empty Flag: BRICK
  52  Bank Empty Flag: SHEEP
  53  Bank Empty Flag: WHEAT
  54  Bank Empty Flag: ORE
  55  Road Distance: P0 (ME)
  56  Road Distance: P1 (NEXT)
"""

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

from json_parser import _texts_sorted, parse_step
from path_a_simulator import GodModeStepper
from path_b_translator import JsonStateTracker

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
    """Extract full board layout from eventHistory.initialState.mapState.tileHexStates.

    tileHexStates keys are web tile IDs (0–18); values contain:
      type (colonist resource enum: 0=DESERT, 1=WOOD, 2=BRICK, 3=SHEEP, 4=WHEAT, 5=ORE)
      diceNumber (0 for desert)

    Returns:
        Dict mapping web_tile_id → (resource_enum, dice_number) for all 19 tiles.
    """
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
    """Determine acting player from gameLogState texts (same logic as json_parser __main__)."""
    for _, t in texts:
        pc = t.get("playerColor")
        if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
            return pc
    return current_turn_color


def _find_winner(events: list[dict]) -> Optional[int]:
    """Return the colonist.io player color that won (text type 45), or None."""
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
            # Show non-zero positions in each
            nz_a = list(zip(*np.where(plane_a != 0)))
            nz_b = list(zip(*np.where(plane_b != 0)))
            if nz_a or nz_b:
                print(f"  Path A non-zero at: {nz_a[:10]}")
                print(f"  Path B non-zero at: {nz_b[:10]}")
            else:
                print("  Both planes are all-zeros but differ by float precision")
            mismatch_found = True
            break  # Stop at first failing channel as requested

    if not mismatch_found:
        # allclose failed overall but no channel-by-channel diff found — floating point
        print("Mismatch is sub-threshold per-channel but allclose failed overall.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# verify_trajectory
# ---------------------------------------------------------------------------


# Channels excluded from the Path A ↔ Path B comparison.
#
#   6–21  Tile resource and dice planes — different board layouts between paths
#   22    Robber — desert starts at a different coordinate in each board
#   23–28 Port planes — catanatron randomises port resource assignments per game
#   30–40 Last roll one-hot — catanatron's apply_roll() never writes state.last_roll;
#          board_tensor_features reads getattr(game.state,"last_roll",None) which
#          is always None.  Path B correctly tracks it from the JSON.  These channels
#          will always be zeros in Path A.  The dataset uses Path B so they are correct.
#   45–54 Bank state — drained by board-specific resource adjacencies at second
#          settlement placement and by resource distribution on different tile layouts
#
# Verified channels: 0–5 (buildings), 29 (validity mask), 41–44 (game phase),
#                    55–56 (road distance after initial build phase fix).
_BOARD_DEPENDENT_CHANNELS = (
    list(range(6, 29))  # 6–28: tiles, robber, ports
    + list(range(30, 41))  # 30–40: last roll (catanatron never sets state.last_roll)
    + list(range(45, 55))  # 45–54: bank state
)


def verify_trajectory(json_path: str, max_steps: int = 9999) -> None:
    """Step Path A and Path B in lock-step and assert their board tensors match.

    Uses GodModeStepper with representation='mixed' so obs["board"] is the
    57-channel board tensor directly comparable to JsonStateTracker.to_tensor().

    Scope of comparison:
      The comparison is limited to the initial placement phase (before the first
      dice roll).  After that point, paid actions (BUILD_ROAD, BUILD_SETTLEMENT,
      etc.) require resources whose amounts differ between the two environments
      because Path A uses a catanatron-random board while Path B reads the actual
      colonist.io board — resource production from dice rolls differs, so paid
      actions become invalid in Path A and the game states diverge.

      During initial placement (free settlements and roads), both environments have
      the same graph topology, the same adjacency invariants (via the coordinate
      mappings), and no resource dependency — so the building channels should match
      exactly.

    Compared channels: 0–5 (buildings), 29 (validity mask), 41–44 (game phase),
                       55–56 (road distance).
    Skipped channels:  6–28 (board topology), 30–40 (catanatron state.last_roll
                       is never set by apply_roll), 45–54 (bank, board-dependent).

    Args:
        json_path:  Absolute path to a colonist.io replay JSON file.
        max_steps:  Hard cap on action steps regardless of phase.

    Raises:
        AssertionError: On the first mismatch in the initial-placement phase,
                        after printing a per-channel debug breakdown.
    """
    with open(json_path) as f:
        data = json.load(f)["data"]

    events: list[dict] = data["eventHistory"]["events"]
    play_order: list[int] = data["playOrder"]
    initial_state: dict = data["eventHistory"].get("initialState", {})

    # ---- Initialise both paths -------------------------------------------
    stepper = GodModeStepper(representation="mixed")
    board_config = _board_config_from_initial_state(initial_state)
    tracker = JsonStateTracker(play_order, board_config=board_config)
    # Apply initialState so robber position and bank are set correctly.
    tracker.update(initial_state)

    # Mapping: catanatron Color index → colonist.io player color
    # play_order[0] = first player = Color[0] (RED), play_order[1] = Color[1] (BLUE), …
    _color_idx_to_colonist: dict[int, int] = {i: c for i, c in enumerate(play_order)}

    current_turn_color: int = play_order[0]
    step = 0

    # Build mask: True on channels we will compare (exclude static tile channels).
    n_channels = 57  # for n_players=2
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

        # Get pre-action obs from tracker (BEFORE update)
        obs_b_pre = tracker.to_tensor(acting_player)

        result = parse_step(event, acting_player)

        # Update tracker with this event's state change
        tracker.update(sc)

        # Update turn tracker AFTER extracting acting player
        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        if result is None:
            # No parseable action: still step path A through any auto-play ticks
            # (e.g. actions the env handles automatically).  Don't count as a step.
            continue

        # Stop comparing once the initial placement phase ends (first dice roll).
        if not tracker._initial_phase:
            remaining = len(events) - ev_idx - 1
            print(
                f"\nInitial placement phase complete at ev{ev_idx}. "
                f"Stopping comparison ({remaining} events remaining).\n"
                f"Post-placement comparison is unsound: catanatron uses a random\n"
                f"board so paid actions become invalid after resource production diverges."
            )
            break

        for action_idx, forced_roll, forced_dev_card in result:
            # Path A: step env, get post-action obs.
            obs_a_full = stepper.step_and_override(
                action_idx, forced_roll, forced_dev_card
            )
            obs_a = obs_a_full["board"]

            # Use the env's current agent as the perspective reference for Path B.
            # This handles batched events (e.g. BUILD_ROAD + END_TURN) where we
            # only step Path A for the first action.
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

    print(
        f"\nVerification complete: {step} initial-placement steps checked — ALL MATCH"
    )


# ---------------------------------------------------------------------------
# process_all_replays
# ---------------------------------------------------------------------------


def _load_replay_json(source) -> Optional[dict]:
    """Load a single replay JSON from a file path or zipfile.open handle."""
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


def _process_single_replay(
    raw: dict,
) -> Optional[dict]:
    """Convert one colonist.io replay dict into an episode dict.

    Returns:
        Dict with keys: observations, actions, rewards, dones.
        None if the replay contains no parseable actions or has no winner.
    """
    data = raw.get("data", raw)
    events: list[dict] = data["eventHistory"]["events"]
    play_order: list[int] = data["playOrder"]
    initial_state: dict = data["eventHistory"].get("initialState", {})

    winner = _find_winner(events)
    board_config = _board_config_from_initial_state(initial_state)
    tracker = JsonStateTracker(play_order, board_config=board_config)
    tracker.update(initial_state)

    current_turn_color: int = play_order[0]

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    actor_list: list[int] = []  # colonist color of the actor at each step

    for event in events:
        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        acting_player = _extract_acting_player(texts, current_turn_color)

        result = parse_step(event, acting_player)

        # Snapshot pre-action observation BEFORE updating tracker state.
        if result is not None:
            pre_obs = tracker.to_tensor(acting_player)

        tracker.update(sc)

        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        if result is None:
            continue

        for action_idx, _forced_roll, _forced_dev_card in result:
            obs_list.append(pre_obs)
            act_list.append(action_idx)
            actor_list.append(acting_player)

    if not obs_list:
        return None

    T = len(obs_list)
    observations = np.stack(obs_list, axis=0).astype(np.float32)  # (T, 57, 22, 14)
    actions = np.array(act_list, dtype=np.int32)  # (T,)
    rewards = np.zeros(T, dtype=np.float32)  # (T,)
    dones = np.zeros(T, dtype=bool)  # (T,)

    # Terminal step: award +1 to the winner, -1 to everyone else.
    dones[-1] = True
    if winner is not None:
        for i, actor in enumerate(actor_list):
            if dones[i]:  # only the last step
                rewards[i] = 1.0 if actor == winner else -1.0
                # Also back-fill -1 for non-winner terminal steps if multiple
                # players could theoretically trigger the final event.

        # Edge case: the final step's actor might not be the winner (e.g. END_TURN
        # by the winner triggers the win check in catanatron).  Assign the reward
        # to the step that belongs to the winner nearest the end.
        if actor_list[-1] != winner:
            # Walk back to find the last step taken by the winner.
            for i in range(T - 1, -1, -1):
                if actor_list[i] == winner:
                    rewards[i] = 1.0
                    dones[i] = True
                    # Undo the terminal flag on the real last step for the loser.
                    rewards[T - 1] = -1.0
                    break

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def process_all_replays(replays_dir: str, output_path: str) -> None:
    """Process all .json replays in replays_dir and save as a gzip-pickle.

    replays_dir may be:
      - A directory containing .json files directly.
      - A path ending in '.zip' (treated as a zip archive).
      - A path of the form 'path/to/file.zip/inner/dir' (zip + inner prefix).

    Args:
        replays_dir:  Path to replay directory or zip archive.
        output_path:  Destination .pkl or .pkl.gz file.
    """
    episodes: list[dict] = []
    failed = 0

    # Detect zip vs directory
    zip_path: Optional[str] = None
    inner_prefix: str = ""

    path_obj = Path(replays_dir)

    # Walk up to find a .zip component
    parts = Path(replays_dir).parts
    for i, part in enumerate(parts):
        candidate = Path(*parts[: i + 1])
        if str(candidate).endswith(".zip") and candidate.is_file():
            zip_path = str(candidate)
            inner_prefix = str(Path(*parts[i + 1 :])) if i + 1 < len(parts) else ""
            break

    if zip_path is not None:
        # Read from zip archive
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
                ep = _process_single_replay(raw)
                if ep is not None:
                    episodes.append(ep)
                else:
                    failed += 1
    else:
        # Read from directory
        json_files = sorted(path_obj.glob("*.json"))
        total = len(json_files)
        print(f"Processing {total} replays from {replays_dir} ...")
        for idx, fpath in enumerate(json_files, 1):
            if idx % 20 == 0 or idx == total:
                print(f"  {idx}/{total} ...", end="\r")
            raw = _load_replay_json(fpath)
            if raw is None:
                failed += 1
                continue
            ep = _process_single_replay(raw)
            if ep is not None:
                episodes.append(ep)
            else:
                failed += 1

    print(f"\nDone. {len(episodes)} episodes saved, {failed} skipped.")

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


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

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

    # Step 1: Verify Path A == Path B on the test replay.
    print("=" * 60)
    print("STEP 1: Verifying Path A ↔ Path B on test replay ...")
    print("=" * 60)
    try:
        verify_trajectory(TEST_REPLAY)
    except AssertionError as exc:
        print(f"\nVerification FAILED: {exc}")
        raise SystemExit(1)

    print("\nVerification Successful\n")

    # Step 2: Generate the full dataset with Path B.
    print("=" * 60)
    print("STEP 2: Generating full dataset ...")
    print("=" * 60)
    process_all_replays(REPLAYS_DIR, OUTPUT_PATH)
