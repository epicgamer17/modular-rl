"""God-mode simulator for offline imitation learning from colonist.io replays.

GodModeStepper wraps the CatanAECEnv and intercepts two stochastic events
before each env.step(), forcing the environment's internal RNG to match the
human game log exactly:

  1. Dev card draw (BUY_DEVELOPMENT_CARD):
       Rotates `game.state.development_listdeck` so the target card is last.
       apply_action.apply_buy_development_card() calls .pop(), so this
       guarantees the exact card the human drew is picked.

  2. Dice roll (ROLL):
       Patches `catanatron.apply_action.roll_dice` with a mock returning
       (a, b) where a + b == forced_roll. The patch is applied only for the
       duration of env.step(), leaving no global state.
"""

import sys
import os
from typing import Optional
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "custom_gym_envs_pkg"))
sys.path.insert(
    0,
    "/Users/jonathanlamontange-kratz/Documents/catanatron-master/catanatron",
)

from custom_gym_envs.envs.catan import env as catan_env_factory


def _dice_pair(forced_roll: int) -> tuple[int, int]:
    """Return a valid (die1, die2) pair that sums to forced_roll.

    Both dice are in [1, 6]. Strategy: maximise die1, then solve for die2.
    Examples: 2→(1,1), 7→(6,1), 11→(6,5), 12→(6,6).
    """
    d1 = max(1, min(forced_roll - 1, 6))
    d2 = forced_roll - d1
    assert 1 <= d2 <= 6, f"Cannot represent roll {forced_roll} as valid dice pair"
    return (d1, d2)


class GodModeStepper:
    """Steps the CatanAECEnv while overriding stochastic RNG events.

    Usage::

        stepper = GodModeStepper()
        obs = stepper.step_and_override(action_idx, forced_roll=None, forced_dev_card=None)
    """

    def __init__(self, **env_kwargs) -> None:
        self.env = catan_env_factory(**env_kwargs)
        self.env.reset()

    def step_and_override(
        self,
        action_idx: int,
        forced_roll: Optional[int],
        forced_dev_card: Optional[str],
    ) -> np.ndarray:
        """Execute one action with optional RNG pre-injection.

        Pre-injection happens before env.step() so the natural game logic
        picks up the forced values without any post-hoc state patching.

        Args:
            action_idx:      Integer index into ACTIONS_ARRAY (0–289).
            forced_roll:     Dice sum (2–12) to force, or None.
            forced_dev_card: Dev card name ('KNIGHT', etc.) to force, or None.

        Returns:
            The 'observation' array from env.last() after the step.
        """
        game = self.env.unwrapped.game

        # ------------------------------------------------------------------ #
        # Pre-injection 1: Dev card deck manipulation                         #
        # Ensures development_listdeck.pop() returns exactly forced_dev_card. #
        # ------------------------------------------------------------------ #
        if forced_dev_card is not None:
            deck = game.state.development_listdeck
            deck.remove(forced_dev_card)   # remove first occurrence
            deck.append(forced_dev_card)   # push to end → .pop() picks it up

        # ------------------------------------------------------------------ #
        # Pre-injection 2: Dice roll mock                                     #
        # Patches roll_dice only for the duration of this single env.step(). #
        # ------------------------------------------------------------------ #
        if forced_roll is not None:
            dice = _dice_pair(forced_roll)
            mock_roll = lambda: dice  # noqa: E731
            with patch("catanatron.apply_action.roll_dice", mock_roll):
                self.env.step(action_idx)
        else:
            self.env.step(action_idx)

        obs_dict, _rew, _term, _trunc, _info = self.env.last()
        return obs_dict["observation"]


# ---------------------------------------------------------------------------
# __main__: dry-run on ZxFhBPVr4KvuC3yN.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Import json_parser from the same package.
    from offline_data.json_parser import parse_step, _texts_sorted

    REPLAY_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "experiments",
        "rainbowzero",
        "catan",
        "replays",
        "ZxFhBPVr4KvuC3yN.json",
    )

    with open(REPLAY_PATH) as f:
        data = json.load(f)

    d = data["data"]
    events = d["eventHistory"]["events"]
    play_order: list[int] = d["playOrder"]

    stepper = GodModeStepper()
    current_turn_color: int = play_order[0]

    step_count = 0
    TARGET = 30

    print(
        f"{'#':<4}  {'ev':<5}  {'action_idx':<12}  "
        f"{'forced_roll':<13}  {'forced_dev_card':<22}  "
        f"{'obs_shape':<18}  obs_sum"
    )
    print("-" * 105)

    for ev_idx, event in enumerate(events):
        if step_count >= TARGET:
            break

        sc = event.get("stateChange", {})
        gls = sc.get("gameLogState", {})
        texts = _texts_sorted(gls)

        # Determine acting player from text (same logic as json_parser __main__).
        acting_player = current_turn_color
        for _, t in texts:
            pc = t.get("playerColor")
            if pc is not None and t.get("type") in (1, 4, 5, 10, 11, 20, 55, 116):
                acting_player = pc
                break

        # Update current_turn_color from currentState after locking acting_player.
        cs = sc.get("currentState", {})
        if "currentTurnPlayerColor" in cs:
            current_turn_color = cs["currentTurnPlayerColor"]

        result = parse_step(event, acting_player)
        if result is None:
            continue

        for action_idx, forced_roll, forced_dev_card in result:
            try:
                obs = stepper.step_and_override(action_idx, forced_roll, forced_dev_card)
                print(
                    f"{step_count:<4}  ev{ev_idx:<4}  {action_idx:<12}  "
                    f"{str(forced_roll):<13}  {str(forced_dev_card):<22}  "
                    f"{str(obs.shape):<18}  {obs.sum():.2f}"
                )
            except Exception as exc:
                print(
                    f"{step_count:<4}  ev{ev_idx:<4}  {action_idx:<12}  "
                    f"{str(forced_roll):<13}  {str(forced_dev_card):<22}  "
                    f"ERROR: {exc}"
                )
            step_count += 1
            if step_count >= TARGET:
                break
