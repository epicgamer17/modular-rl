"""
TicTacToe Expert Component
===========================
A pipeline component that wraps the minimax-heuristic TicTacToe expert.

The heuristic is a direct port of the logic from the retired
``actors/experts/tictactoe_expert.py``.  It prefers winning moves,
then blocking moves, then falls back to a random legal action – all in
a fully vectorised NumPy style (no Python loops over the action candidates).

Blackboard contract
-------------------
Reads:
    ``data["obs"]`` – torch.Tensor or np.ndarray of shape ``[1, 2, 3, 3]``
        or ``[2, 3, 3]``.  Plane 0 is the current player's pieces,
        plane 1 the opponent's.  The observation format matches the
        PettingZoo ``tictactoe_v3`` environment wrapped with
        ``ActionMaskInInfoWrapper``.
    ``data["info"]`` – Dict with key ``"legal_moves"`` (``List[int]``).

Writes:
    ``meta["action"]`` – ``int`` selected cell index in ``[0, 9)``.
"""

import random
import numpy as np
import torch
from typing import Optional

from core import PipelineComponent, Blackboard

# Board shape constants.
_BOARD_ROWS = 3
_BOARD_COLS = 3
_NUM_CELLS = _BOARD_ROWS * _BOARD_COLS  # 9


def _select_tictactoe_action(board: np.ndarray, legal_moves: list) -> int:
    """
    Apply the minimax heuristic to choose an action on a TicTacToe board.

    Priority:
    1. Win immediately (complete a row / col / diagonal of the current player).
    2. Block the opponent from winning (complete their row / col / diagonal).
    3. Fall back to a random legal action.

    Args:
        board:       2-D float array of shape ``(3, 3)``.
                     +1 ≡ current player, -1 ≡ opponent, 0 ≡ empty.
        legal_moves: List of legal cell indices (flat, row-major).

    Returns:
        Integer cell index in ``[0, 9)``.
    """
    assert board.shape == (_BOARD_ROWS, _BOARD_COLS), (
        f"_select_tictactoe_action: expected board shape ({_BOARD_ROWS}, {_BOARD_COLS}), "
        f"got {board.shape}."
    )
    assert len(legal_moves) > 0, (
        "_select_tictactoe_action: legal_moves is empty; cannot choose an action."
    )

    # Default: random legal move (priority 3).
    action: int = int(random.choice(legal_moves))
    # Candidate block move (will be overwritten by win if both are found).
    block_action: Optional[int] = None

    # --- Rows ---
    for row in range(_BOARD_ROWS):
        line = board[row, :]
        if np.sum(line) == 2 and 0 in line:           # current player wins
            col = int(np.where(line == 0)[0][0])
            return int(np.ravel_multi_index((row, col), (_BOARD_ROWS, _BOARD_COLS)))
        if abs(np.sum(line)) == 2 and 0 in line:      # must block opponent
            col = int(np.where(line == 0)[0][0])
            block_action = int(np.ravel_multi_index((row, col), (_BOARD_ROWS, _BOARD_COLS)))

    # --- Columns ---
    for col in range(_BOARD_COLS):
        line = board[:, col]
        if np.sum(line) == 2 and 0 in line:
            row = int(np.where(line == 0)[0][0])
            return int(np.ravel_multi_index((row, col), (_BOARD_ROWS, _BOARD_COLS)))
        if abs(np.sum(line)) == 2 and 0 in line:
            row = int(np.where(line == 0)[0][0])
            block_action = int(np.ravel_multi_index((row, col), (_BOARD_ROWS, _BOARD_COLS)))

    # --- Main diagonal ---
    diag = board.diagonal()
    if np.sum(diag) == 2 and 0 in diag:
        idx = int(np.where(diag == 0)[0][0])
        return int(np.ravel_multi_index((idx, idx), (_BOARD_ROWS, _BOARD_COLS)))
    if abs(np.sum(diag)) == 2 and 0 in diag:
        idx = int(np.where(diag == 0)[0][0])
        block_action = int(np.ravel_multi_index((idx, idx), (_BOARD_ROWS, _BOARD_COLS)))

    # --- Anti-diagonal ---
    anti_diag = np.fliplr(board).diagonal()
    if np.sum(anti_diag) == 2 and 0 in anti_diag:
        idx = int(np.where(anti_diag == 0)[0][0])
        return int(np.ravel_multi_index((idx, _BOARD_COLS - 1 - idx), (_BOARD_ROWS, _BOARD_COLS)))
    if abs(np.sum(anti_diag)) == 2 and 0 in anti_diag:
        idx = int(np.where(anti_diag == 0)[0][0])
        block_action = int(
            np.ravel_multi_index((idx, _BOARD_COLS - 1 - idx), (_BOARD_ROWS, _BOARD_COLS))
        )

    # Return block move if a win wasn't found; otherwise fall back to random.
    return block_action if block_action is not None else action


class TicTacToeExpertComponent(PipelineComponent):
    """
    Pipeline component wrapping the TicTacToe minimax heuristic.

    Reads the current observation from the Blackboard, reconstructs the
    logical board state, applies the win-first / block-second heuristic,
    and writes the chosen action to ``blackboard.meta["action"]``.

    The component is stateless and can be used for any player seat in a
    ``PlayerRoutingComponent``.

    Example pipeline usage::

        pipeline = BlackboardEngine(components=[
            PettingZooObservationComponent(env),
            TicTacToeExpertComponent(),
            PettingZooStepComponent(env, obs_component),
        ])
    """

    def execute(self, blackboard: Blackboard) -> None:
        """
        Compute and publish the expert's chosen action.

        Args:
            blackboard: The shared Blackboard for the current pipeline tick.

        Raises:
            AssertionError: If ``blackboard.data["obs"]`` is None or has
                an unexpected shape, or if ``legal_moves`` is empty.
        """
        obs = blackboard.data.get("obs")
        assert obs is not None, (
            "TicTacToeExpertComponent: 'obs' is missing from blackboard.data. "
            "An observation component must run before this expert."
        )

        # Convert to NumPy if needed.
        if torch.is_tensor(obs):
            obs_np: np.ndarray = obs.cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        # Strip optional batch dimension: [1, 2, 3, 3] → [2, 3, 3].
        if obs_np.ndim == 4:
            obs_np = obs_np[0]

        # Handle frame stacking: take the 2 most recent planes.
        # Observation is [2*k + 1, 3, 3]. Planes 0,1 are the current frame.
        if obs_np.shape[0] > 2:
            obs_np = obs_np[:2]

        assert obs_np.ndim == 3 and obs_np.shape == (2, _BOARD_ROWS, _BOARD_COLS), (
            f"TicTacToeExpertComponent: expected obs shape (2, 3, 3) or (1, 2, 3, 3), "
            f"got {obs_np.shape}."
        )

        # Reconstruct the logical board: +1 current player, -1 opponent.
        board: np.ndarray = obs_np[0] - obs_np[1]  # (3, 3)

        info: dict = blackboard.data.get("info", {})
        if info is None:
            info = {}
        legal_moves: list = info.get("legal_moves", list(range(_NUM_CELLS)))

        if len(legal_moves) == 0:
            legal_moves = list(range(_NUM_CELLS))

        action: int = _select_tictactoe_action(board, legal_moves)
        blackboard.meta["action"] = action


class TicTacToeBestAgent:
    """
    Legacy compatibility adapter for code that still calls
    ``agent.predict(obs, info)`` → ``agent.select_actions(prediction, info)``.

    This class preserves the old ``actors.experts.tictactoe_expert`` interface
    so that ``Tester`` and any other callers that have not yet been migrated to
    the component model continue to work unmodified.

    New code should use ``TicTacToeExpertComponent`` instead.

    Args:
        name: Optional display name (unused internally).
    """

    def __init__(self, name: str = "tictactoe_expert") -> None:
        self.name = name

    def predict(self, observation: np.ndarray, info: dict, env=None, *args, **kwargs):
        """
        Pass-through prediction step.

        Args:
            observation: Raw observation array.
            info:        Info dict (may contain ``"legal_moves"``).
            env:         Unused; kept for API compatibility.

        Returns:
            Tuple of ``(observation, info)`` unchanged.
        """
        return observation, info

    def select_actions(self, prediction, info: dict, *args, **kwargs) -> int:
        """
        Apply the minimax heuristic and return the chosen action index.

        The observation is read from ``prediction[0]`` when *prediction* is a
        tuple/list (matching the return value of ``predict()``).  The method
        handles both ``(2, 3, 3)`` and ``(1, 2, 3, 3)`` array shapes.

        Args:
            prediction: Value returned by ``predict()`` – either a NumPy
                array or a ``(obs, info)`` tuple.
            info:       Info dict containing ``"legal_moves"``.

        Returns:
            Integer flat cell index in ``[0, 9)``.
        """
        obs = prediction[0] if isinstance(prediction, (tuple, list)) else prediction

        obs_np: np.ndarray = np.asarray(obs, dtype=np.float32)

        # Strip optional batch dimension: [1, 2, 3, 3] → [2, 3, 3]
        if obs_np.ndim == 4:
            obs_np = obs_np[0]

        # Handle frame stacking: take the 2 most recent planes.
        # Observation is [2*k + 1, 3, 3]. Planes 0,1 are the current frame.
        if obs_np.shape[0] > 2:
            obs_np = obs_np[:2]

        assert obs_np.ndim == 3 and obs_np.shape == (2, _BOARD_ROWS, _BOARD_COLS), (
            f"TicTacToeBestAgent.select_actions: expected obs shape (2, 3, 3) or "
            f"(1, 2, 3, 3), got {obs_np.shape}."
        )

        board: np.ndarray = obs_np[0] - obs_np[1]  # (3, 3)

        legal_moves: list = info.get("legal_moves", list(range(_NUM_CELLS)))
        if not legal_moves:
            legal_moves = list(range(_NUM_CELLS))

        return _select_tictactoe_action(board, legal_moves)
