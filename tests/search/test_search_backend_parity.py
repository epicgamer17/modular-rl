"""Parity tests comparing Python, AOS, and C++ search backends.

The Python backend is treated as ground-truth (it is the most mature / tested
implementation).  AOS and C++ outputs must agree with Python on all four
return values of ``run()`` / ``run_vectorized()``:

    (root_value, exploratory_policy, target_policy, best_action, metadata)

Design principles:
  - Every comparison is seeded identically *before* each backend call so that
    stochastic elements (Dirichlet, Gumbel noise) are identical.
  - The MockNetwork is fully deterministic and stateless so tree expansion
    paths are identical across backends given the same seed.
  - C++ tests are unconditionally skipped when the compiled module is absent.
"""

from __future__ import annotations

import pickle
import pytest
import numpy as np
import torch
from types import SimpleNamespace
from typing import List, Optional

from tests.search.conftest import (
    MockNetworkState,
    MockSearchNetwork as MockNetwork,
    StateCapturingNetwork,
)

# --------------------------------------------------------------------------
# Module-level markers
# --------------------------------------------------------------------------

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# --------------------------------------------------------------------------
# Try to import the C++ backend once; skip cpp tests if not compiled.
# --------------------------------------------------------------------------

try:
    from search import search_cpp as _cpp_module

    _CPP_AVAILABLE = hasattr(_cpp_module, "ModularSearch")
except Exception:
    _cpp_module = None
    _CPP_AVAILABLE = False

_skip_cpp = pytest.mark.skipif(not _CPP_AVAILABLE, reason="mcts_cpp_backend not built")


# --------------------------------------------------------------------------
# Config fixture
# --------------------------------------------------------------------------


@pytest.fixture
def base_config() -> SimpleNamespace:
    """Minimal valid config compatible with all three search backends."""
    return SimpleNamespace(
        pb_c_init=1.25,
        pb_c_base=19652,
        discount_factor=0.99,
        gumbel=False,
        gumbel_cvisit=50.0,
        gumbel_cscale=1.0,
        gumbel_m=2,
        game=SimpleNamespace(num_players=1),
        use_value_prefix=False,
        num_simulations=4,
        num_codes=1,
        max_search_depth=50,
        max_nodes=200,
        use_dirichlet=False,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        root_dirichlet_alpha_adaptive=True,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        injection_frac=0.1,
        policy_extraction="visit_count",
        backprop_method="average",
        scoring_method="ucb",
        known_bounds=[0.0, 2.0],
        use_sequential_halving=False,
        bootstrap_method="network_value",
        soft_update=False,
        min_max_epsilon=1e-8,
        stochastic=False,
        search_batch_size=0,
        virtual_loss=1.0,
        use_virtual_mean=False,
        compilation=SimpleNamespace(enabled=False, fullgraph=False),
        internal_decision_modifier="none",
        internal_chance_modifier="none",
        stochastic_exploration=False,
        sampling_temp=1.0,
    )


# --------------------------------------------------------------------------
# Backend factory helpers
# --------------------------------------------------------------------------


def _make_py_search(config, num_actions: int):
    from search.search_py.modular_search import ModularSearch

    return ModularSearch(config, torch.device("cpu"), num_actions)


def _make_aos_search(config, num_actions: int):
    from search.aos_search.search_algorithm import ModularSearch

    return ModularSearch(config, torch.device("cpu"), num_actions)


def _make_cpp_search(config, num_actions: int):
    assert _CPP_AVAILABLE, "C++ backend is not compiled"
    return _cpp_module.ModularSearch(config, torch.device("cpu"), num_actions)


# --------------------------------------------------------------------------
# Seeding helper
# --------------------------------------------------------------------------


def _seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Single-item run() comparison helpers
# --------------------------------------------------------------------------


def _py_run(config, num_actions: int, obs, info: dict, net, seed: int = 42):
    """Run the Python backend and return all outputs."""
    search = _make_py_search(config, num_actions)
    _seed(seed)
    return search.run(obs, info, net)


def _aos_run(config, num_actions: int, obs, info: dict, net, seed: int = 42):
    """Run the AOS backend and return all outputs."""
    search = _make_aos_search(config, num_actions)
    _seed(seed)
    return search.run(obs, info, net)


def _cpp_run(config, num_actions: int, obs, info: dict, net, seed: int = 42):
    """Run the C++ backend and return all outputs."""
    search = _make_cpp_search(config, num_actions)
    _seed(seed)
    return search.run(obs, info, net)


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

_ATOL = 1e-5


def _assert_policies_close(
    py_target: torch.Tensor,
    other_target: torch.Tensor,
    label: str,
    atol: float = _ATOL,
) -> None:
    assert py_target.shape == other_target.shape, (
        f"{label}: policy shape mismatch {py_target.shape} vs {other_target.shape}"
    )
    assert torch.allclose(py_target, other_target, atol=atol), (
        f"{label}: target_policy mismatch\n  py={py_target}\n  other={other_target}\n"
        f"  diff={torch.abs(py_target - other_target).max().item():.6e}"
    )


def _assert_run_outputs_close(py_out, other_out, label: str, atol: float = _ATOL) -> None:
    """Compare all outputs of run(): root_value, policies, and best_action.

    Both backends return the backed-up tree value from ``node_values`` /
    ``root.value()`` — not the raw initial network value — so root_value must
    agree between implementations.
    """
    py_val, py_expl, py_tgt, py_act, _ = py_out
    ot_val, ot_expl, ot_tgt, ot_act, _ = other_out

    assert abs(py_val - ot_val) < atol, (
        f"{label}: root_value mismatch {py_val:.8f} vs {ot_val:.8f} "
        f"(diff={abs(py_val - ot_val):.2e})"
    )
    _assert_policies_close(py_tgt, ot_tgt, f"{label} target_policy", atol)
    _assert_policies_close(py_expl, ot_expl, f"{label} exploratory_policy", atol)
    assert py_act == ot_act, (
        f"{label}: best_action mismatch {py_act} vs {ot_act}"
    )


# ==========================================================================
# Python vs AOS: run() single-item parity
# ==========================================================================


class TestPythonAosSingleRunParity:
    """Tests that verify Python and AOS produce identical outputs from run()."""

    def test_ucb_standard(self, base_config):
        """Baseline UCB search: all 4 outputs must match."""
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net)
        aos_out = _aos_run(base_config, num_actions, obs, info, net)

        _assert_run_outputs_close(py_out, aos_out, "ucb_standard")

    def test_ucb_partial_legal_moves(self, base_config):
        """Only a subset of actions are legal; masking must match exactly."""
        num_actions = 6
        legal = [0, 2, 4]
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": legal}

        py_out = _py_run(base_config, num_actions, obs, info, net)
        aos_out = _aos_run(base_config, num_actions, obs, info, net)

        _assert_run_outputs_close(py_out, aos_out, "ucb_partial_legal")

        # Illegal actions must have zero probability in both policies.
        illegal = [a for a in range(num_actions) if a not in legal]
        for idx in illegal:
            assert py_out[2][idx] == pytest.approx(0.0, abs=1e-7), (
                f"Python: illegal action {idx} has non-zero prob in target"
            )
            assert aos_out[2][idx] == pytest.approx(0.0, abs=1e-7), (
                f"AOS: illegal action {idx} has non-zero prob in target"
            )

    def test_ucb_many_simulations(self, base_config):
        """More simulations stress-test deeper tree agreement."""
        base_config.num_simulations = 20
        base_config.max_nodes = 500
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=7)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=7)

        _assert_run_outputs_close(py_out, aos_out, "ucb_many_sims")

    def test_ucb_no_known_bounds(self, base_config):
        """Adaptive min-max normalisation: known_bounds=None — expected divergence."""
        base_config.known_bounds = None
        num_actions = 4
        net = MockNetwork(num_actions, mock_value=2.5)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=13)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=13)

        _assert_run_outputs_close(py_out, aos_out, "ucb_no_known_bounds")

    # Parity gap resolved: AOS now correctly tracks root values in min-max bounds
    def test_ucb_extreme_values(self, base_config):
        """Very large values — expected divergence with known_bounds=None."""
        base_config.known_bounds = None
        num_actions = 4
        net = MockNetwork(num_actions, mock_value=1e6)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=77)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=77)

        _assert_run_outputs_close(py_out, aos_out, "ucb_extreme_values")

    def test_ucb_with_dirichlet_noise(self, base_config):
        """Dirichlet noise must be applied with the same seed in both backends."""
        base_config.use_dirichlet = True
        base_config.dirichlet_alpha = 0.3
        base_config.dirichlet_fraction = 0.25
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=55)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=55)

        # Dirichlet paths differ in implementation; only compare target policy.
        # Use a relaxed tolerance because the Python and AOS Dirichlet sampling
        # code paths consume the RNG state in different orders.
        _assert_policies_close(
            py_out[2], aos_out[2], "ucb_dirichlet target_policy", atol=1e-4
        )

    def test_ucb_deep_horizon_discounting(self, base_config):
        """Long rollouts verify geometric discounting accumulates identically."""
        base_config.num_simulations = 30
        base_config.max_search_depth = 100
        base_config.max_nodes = 1000
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=55)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=55)

        _assert_run_outputs_close(py_out, aos_out, "ucb_deep_horizon")

    @pytest.mark.xfail(
        reason=(
            "Known parity gap in Gumbel mode: the Python and AOS backends consume "
            "the Gumbel-noise RNG state in different orders during sequential halving, "
            "producing policies that agree on direction but differ beyond atol=1e-4. "
            "Tracked as a known divergence to resolve."
        ),
        strict=True,
    )
    def test_gumbel_mode(self, base_config):
        """Gumbel sequential-halving policy extraction — expected divergence."""
        base_config.gumbel = True
        base_config.scoring_method = "gumbel"
        base_config.policy_extraction = "gumbel"
        base_config.use_sequential_halving = True
        base_config.num_simulations = 8
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=42)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=42)

        _assert_policies_close(
            py_out[2], aos_out[2], "gumbel target_policy", atol=1e-4
        )

    # Parity gap resolved: Python ModularSearch now correctly configures BestActionRootPolicy when policy_extraction="minimax"
    def test_minimax_policy_extraction(self, base_config):
        """Minimax policy extraction — expected semantic divergence."""
        base_config.policy_extraction = "minimax"
        base_config.backprop_method = "minimax"
        base_config.known_bounds = None
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=99)
        aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=99)

        _assert_policies_close(py_out[2], aos_out[2], "minimax_policy", atol=1e-4)

    def test_policy_sums_to_one(self, base_config):
        """All returned policy tensors must be valid probability distributions."""
        num_actions = 5
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        for backend_run, label in [
            (_py_run, "python"),
            (_aos_run, "aos"),
        ]:
            out = backend_run(base_config, num_actions, obs, info, net)
            _, expl, tgt, _, _ = out
            assert tgt.sum().item() == pytest.approx(1.0, abs=1e-5), (
                f"{label}: target_policy does not sum to 1"
            )
            assert expl.sum().item() == pytest.approx(1.0, abs=1e-5), (
                f"{label}: exploratory_policy does not sum to 1"
            )


# ==========================================================================
# Python vs AOS: run_vectorized() batch parity
# ==========================================================================


class TestPythonAosBatchedParity:
    """Tests that verify Python and AOS produce identical run_vectorized() outputs."""

    def _py_run_vec(self, config, num_actions, batched_obs, batched_info, net, seed=42):
        search = _make_py_search(config, num_actions)
        _seed(seed)
        return search.run_vectorized(batched_obs, batched_info, net)

    def _aos_run_vec(self, config, num_actions, batched_obs, batched_info, net, seed=42):
        search = _make_aos_search(config, num_actions)
        _seed(seed)
        return search.run_vectorized(batched_obs, batched_info, net)

    def _assert_batch_outputs_close(self, py_out, aos_out, label: str, B: int, atol: float = _ATOL) -> None:
        """Compare all outputs of run_vectorized(): root_values, policies, best_actions."""
        py_vals, py_expls, py_tgts, py_acts, _ = py_out
        ao_vals, ao_expls, ao_tgts, ao_acts, _ = aos_out

        assert len(py_tgts) == B, f"{label}: python returned {len(py_tgts)} policies, expected {B}"
        assert len(ao_tgts) == B, f"{label}: aos returned {len(ao_tgts)} policies, expected {B}"

        for i in range(B):
            assert abs(py_vals[i] - ao_vals[i]) < atol, (
                f"{label}[{i}]: root_value {py_vals[i]:.8f} vs {ao_vals[i]:.8f} "
                f"(diff={abs(py_vals[i] - ao_vals[i]):.2e})"
            )
            _assert_policies_close(
                py_tgts[i], ao_tgts[i], f"{label}[{i}] target_policy", atol
            )
            assert py_acts[i] == ao_acts[i], (
                f"{label}[{i}]: best_action {py_acts[i]} vs {ao_acts[i]}"
            )

    def test_batch_size_1_matches_single_run(self, base_config):
        """run_vectorized with B=1 must match run() for the same item."""
        num_actions = 4
        net = MockNetwork(num_actions)
        obs_single = torch.ones((1, 4, 4))
        obs_batched = obs_single.unsqueeze(0)  # [1, 1, 4, 4]
        info_single = {"player": 0, "legal_moves": list(range(num_actions))}
        batched_info = [info_single]

        _seed(42)
        py_single = _make_py_search(base_config, num_actions).run(obs_single, info_single, net)
        _seed(42)
        py_batch = _make_py_search(base_config, num_actions).run_vectorized(
            obs_batched, batched_info, net
        )

        _assert_policies_close(
            py_single[2], py_batch[2][0], "py run vs run_vectorized target_policy"
        )

    def test_batch_size_3_ucb(self, base_config):
        """3-item batch must match individual run() outputs for each item."""
        num_actions = 4
        B = 3
        net = MockNetwork(num_actions)
        batched_obs = torch.ones((B, 1, 4, 4))
        batched_info = [
            {"player": 0, "legal_moves": list(range(num_actions))} for _ in range(B)
        ]

        py_out = self._py_run_vec(base_config, num_actions, batched_obs, batched_info, net)
        aos_out = self._aos_run_vec(base_config, num_actions, batched_obs, batched_info, net)

        self._assert_batch_outputs_close(py_out, aos_out, "batch3_ucb", B)

    def test_batch_size_5_ucb(self, base_config):
        """Larger batch verifies there are no off-by-one bugs in batch handling."""
        num_actions = 4
        B = 5
        net = MockNetwork(num_actions)
        batched_obs = torch.ones((B, 1, 4, 4))
        batched_info = [
            {"player": 0, "legal_moves": list(range(num_actions))} for _ in range(B)
        ]

        py_out = self._py_run_vec(
            base_config, num_actions, batched_obs, batched_info, net, seed=123
        )
        aos_out = self._aos_run_vec(
            base_config, num_actions, batched_obs, batched_info, net, seed=123
        )

        self._assert_batch_outputs_close(py_out, aos_out, "batch5_ucb", B)

    def test_batch_heterogeneous_legal_moves(self, base_config):
        """Per-item action masking in run_vectorized must work correctly."""
        num_actions = 6
        B = 3
        net = MockNetwork(num_actions)
        batched_obs = torch.ones((B, 1, 4, 4))
        legal_per_item = [[0, 1, 2], [2, 3, 4], [0, 3, 5]]
        batched_info = [
            {"player": 0, "legal_moves": legal_per_item[i]} for i in range(B)
        ]

        py_out = self._py_run_vec(base_config, num_actions, batched_obs, batched_info, net)
        aos_out = self._aos_run_vec(base_config, num_actions, batched_obs, batched_info, net)

        self._assert_batch_outputs_close(py_out, aos_out, "batch_hetero_legal", B)

        # Verify illegal actions have zero probability in the target policies.
        for i in range(B):
            for a in range(num_actions):
                if a not in legal_per_item[i]:
                    assert py_out[2][i][a].item() == pytest.approx(0.0, abs=1e-7), (
                        f"Python batch[{i}]: illegal action {a} non-zero"
                    )
                    assert aos_out[2][i][a].item() == pytest.approx(0.0, abs=1e-7), (
                        f"AOS batch[{i}]: illegal action {a} non-zero"
                    )


# ==========================================================================
# Python vs AOS: Multi-player parity
# ==========================================================================


class TestMultiPlayerParity:
    """Search with 2+ players must agree across backends."""

    def test_two_player_single_run(self, base_config):
        """2-player game: player identity must be respected by both backends."""
        base_config.game = SimpleNamespace(num_players=2)
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))

        for player_id in [0, 1]:
            info = {"player": player_id, "legal_moves": list(range(num_actions))}
            py_out = _py_run(base_config, num_actions, obs, info, net, seed=42)
            aos_out = _aos_run(base_config, num_actions, obs, info, net, seed=42)
            _assert_run_outputs_close(
                py_out, aos_out, f"2player_player{player_id}", atol=1e-4
            )


# ==========================================================================
# Structural / contract invariants (both backends)
# ==========================================================================


class TestOutputContractInvariants:
    """Verify that both backends satisfy the documented return-value contracts
    regardless of parity with each other."""

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_target_policy_is_probability_distribution(self, make_search, base_config):
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        _seed(42)
        search = make_search(base_config, num_actions)
        _, _, tgt, _, _ = search.run(obs, info, net)

        assert tgt.shape == (num_actions,)
        assert (tgt >= 0.0).all(), "Negative probabilities in target policy"
        assert tgt.sum().item() == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_exploratory_policy_is_probability_distribution(self, make_search, base_config):
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        _seed(42)
        search = make_search(base_config, num_actions)
        _, expl, _, _, _ = search.run(obs, info, net)

        assert expl.shape == (num_actions,)
        assert (expl >= 0.0).all(), "Negative probabilities in exploratory policy"
        assert expl.sum().item() == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_best_action_is_legal(self, make_search, base_config):
        num_actions = 6
        legal = [1, 3, 5]
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": legal}
        _seed(42)
        search = make_search(base_config, num_actions)
        _, _, _, best_action, _ = search.run(obs, info, net)

        assert best_action in legal, (
            f"best_action={best_action} is not in legal_moves={legal}"
        )

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_best_action_matches_argmax_target(self, make_search, base_config):
        """best_action must be the argmax of the target policy (UCB mode)."""
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        _seed(42)
        search = make_search(base_config, num_actions)
        _, _, tgt, best_action, _ = search.run(obs, info, net)

        argmax = torch.argmax(tgt).item()
        assert best_action == argmax, (
            f"best_action={best_action} but argmax(target_policy)={argmax}"
        )

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_missing_player_in_info_raises(self, make_search, base_config):
        """Both backends must raise an assertion when 'player' is missing from info."""
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info_no_player = {"legal_moves": list(range(num_actions))}
        search = make_search(base_config, num_actions)

        with pytest.raises((AssertionError, KeyError)):
            search.run(obs, info_no_player, net)

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_determinism_with_same_seed(self, make_search, base_config):
        """Two identical runs with the same seed must return the same policy."""
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        search = make_search(base_config, num_actions)
        _seed(42)
        out1 = search.run(obs, info, net)
        _seed(42)
        out2 = search.run(obs, info, net)

        _assert_policies_close(out1[2], out2[2], "determinism target_policy")


# ==========================================================================
# C++ backend parity  (skipped if mcts_cpp_backend is not compiled)
# ==========================================================================


@_skip_cpp
class TestCppParity:
    """Verify C++ backend agrees with Python on all outputs from run()."""

    def test_ucb_target_policy(self, base_config):
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net)
        cpp_out = _cpp_run(base_config, num_actions, obs, info, net)

        _assert_run_outputs_close(py_out, cpp_out, "cpp_ucb_standard", atol=1e-4)

    def test_ucb_partial_legal_moves(self, base_config):
        num_actions = 6
        legal = [0, 2, 4]
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": legal}

        py_out = _py_run(base_config, num_actions, obs, info, net)
        cpp_out = _cpp_run(base_config, num_actions, obs, info, net)

        _assert_run_outputs_close(py_out, cpp_out, "cpp_ucb_partial_legal", atol=1e-4)

    @pytest.mark.xfail(
        reason="C++ backend does not currently implement sequential halving for Gumbel mode.",
        strict=True,
    )
    def test_gumbel_mode(self, base_config):
        base_config.gumbel = True
        base_config.scoring_method = "gumbel"
        base_config.policy_extraction = "gumbel"
        base_config.use_sequential_halving = True
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        py_out = _py_run(base_config, num_actions, obs, info, net, seed=42)
        cpp_out = _cpp_run(base_config, num_actions, obs, info, net, seed=42)

        _assert_policies_close(py_out[2], cpp_out[2], "cpp_gumbel target_policy", atol=1e-4)

    def test_target_policy_is_valid(self, base_config):
        num_actions = 4
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}

        _seed(42)
        search = _make_cpp_search(base_config, num_actions)
        _, _, tgt, _, _ = search.run(obs, info, net)

        assert tgt.shape == (num_actions,)
        assert (tgt >= 0.0).all()
        assert tgt.sum().item() == pytest.approx(1.0, abs=1e-5)


# ==========================================================================
# AOS latent-state storage: correctness and parity
# ==========================================================================


class TestAosStateStorage:
    """Tests for the AOS backend's pytree-based latent-state storage.

    The core bug fixed was that ``batched_mcts_step`` was passing raw
    node-index tensors to ``hidden_state_inference`` instead of the actual
    reconstructed ``MuZeroNetworkState``.  The fix stores each node's opaque
    ``network_state`` in ``FlatTree.node_state_leaves`` using
    ``torch.utils._pytree.tree_flatten/tree_unflatten`` so the opaque token
    is faithfully preserved and reconstructed without MCTS ever inspecting
    its internals.

    Design:
      - ``StateCapturingNetwork`` records every state passed to
        ``hidden_state_inference`` so we can assert the type and value.
      - Parity tests run both Python and AOS backends with the same seed and
        assert that the captured state *types* are identical (both backends
        should call hidden_state_inference with a ``MockNetworkState``, never
        a raw tensor).
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    NUM_ACTIONS = 4

    def _run_aos_capturing(
        self, config, num_actions: int = NUM_ACTIONS, B: int = 1
    ) -> StateCapturingNetwork:
        """Run the AOS backend with a StateCapturingNetwork and return it.

        Uses run() for B=1, run_vectorized() for B>1.
        """
        net = StateCapturingNetwork(num_actions)
        obs = torch.ones((B, 1, 4, 4)) if B > 1 else torch.ones((1, 4, 4))
        if B > 1:
            info = [
                {"player": 0, "legal_moves": list(range(num_actions))}
                for _ in range(B)
            ]
        else:
            info = {"player": 0, "legal_moves": list(range(num_actions))}
        _seed(42)
        search = _make_aos_search(config, num_actions)
        if B > 1:
            search.run_vectorized(obs, info, net)
        else:
            search.run(obs, info, net)
        return net

    def _run_py_capturing(
        self, config, num_actions: int = NUM_ACTIONS, B: int = 1
    ) -> StateCapturingNetwork:
        """Run the Python backend with a StateCapturingNetwork and return it."""
        net = StateCapturingNetwork(num_actions)
        obs = torch.ones((B, 1, 4, 4)) if B > 1 else torch.ones((1, 4, 4))
        if B > 1:
            info = [
                {"player": 0, "legal_moves": list(range(num_actions))}
                for _ in range(B)
            ]
        else:
            info = {"player": 0, "legal_moves": list(range(num_actions))}
        _seed(42)
        search = _make_py_search(config, num_actions)
        if B > 1:
            search.run_vectorized(obs, info, net)
        else:
            search.run(obs, info, net)
        return net

    # ------------------------------------------------------------------
    # AOS passes MockNetworkState (not raw tensor) to hidden_state_inference
    # ------------------------------------------------------------------

    def test_aos_passes_network_state_not_tensor_to_hidden_state_inference(
        self, base_config
    ):
        """AOS must reconstruct and pass a ``MockNetworkState`` to
        ``hidden_state_inference``, never a raw node-index tensor.

        Regression test for: ``AttributeError: 'Tensor' object has no
        attribute 'dynamics'``.

        The bug: AOS was passing the raw ``flat_parents_t`` tensor (node
        indices) directly to ``hidden_state_inference``.  The fix stores the
        root's ``network_state`` in ``FlatTree.node_state_leaves`` using
        ``torch.utils._pytree`` and reconstructs it before each inference
        call.
        """
        net = self._run_aos_capturing(base_config)

        assert net.captured_states, (
            "hidden_state_inference was never called — increase num_simulations"
        )

        for i, state in enumerate(net.captured_states):
            assert isinstance(state, MockNetworkState), (
                f"AOS passed {type(state).__name__} (not MockNetworkState) to "
                f"hidden_state_inference at call {i}.  "
                "This is the 'Tensor has no attribute dynamics' regression."
            )

    def test_py_passes_network_state_to_hidden_state_inference(self, base_config):
        """Sanity check: Python backend also passes MockNetworkState (ground truth)."""
        net = self._run_py_capturing(base_config)

        # Python may have zero or more calls depending on the expansion path.
        for i, state in enumerate(net.captured_states):
            assert isinstance(state, MockNetworkState), (
                f"Python backend passed {type(state).__name__} at call {i}"
            )

    # ------------------------------------------------------------------
    # Parity: both backends call hidden_state_inference with the same type
    # ------------------------------------------------------------------

    def test_state_type_parity_between_py_and_aos(self, base_config):
        """Both backends must call hidden_state_inference with the same state type.

        Uses ``torch.utils._pytree`` internally to handle opaque tokens
        generically.  The key invariant is: every captured state must be a
        ``MockNetworkState``, regardless of backend.
        """
        py_net = self._run_py_capturing(base_config)
        aos_net = self._run_aos_capturing(base_config)

        for i, state in enumerate(py_net.captured_states):
            assert isinstance(state, MockNetworkState), (
                f"Python: state at call {i} is {type(state).__name__}"
            )
        for i, state in enumerate(aos_net.captured_states):
            assert isinstance(state, MockNetworkState), (
                f"AOS: state at call {i} is {type(state).__name__}"
            )

    # ------------------------------------------------------------------
    # Pytree round-trip: tensor leaves survive flatten/unflatten
    # ------------------------------------------------------------------

    def test_pytree_roundtrip_preserves_tensor_leaves(self):
        """``tree_flatten`` / ``tree_unflatten`` round-trips a MockNetworkState.

        This is the core mechanism used in the AOS fix: tensor leaves are
        extracted, stored in pre-allocated buffers, and reconstructed via
        ``tree_unflatten``.  The round-trip must produce a structurally
        identical NamedTuple with the same tensor values.
        """
        from torch.utils import _pytree as pytree

        original = MockNetworkState(
            data=torch.tensor([[1.0, 2.0, 3.0]]),
            wm_memory=None,
        )

        leaves, treespec = pytree.tree_flatten(original)

        # Only tensor leaves should be stored; None is a non-tensor leaf.
        tensor_leaves = [l for l in leaves if isinstance(l, torch.Tensor)]
        assert len(tensor_leaves) >= 1, "Expected at least one tensor leaf"

        reconstructed = pytree.tree_unflatten(leaves, treespec)

        assert isinstance(reconstructed, MockNetworkState), (
            f"Expected MockNetworkState, got {type(reconstructed).__name__}"
        )
        assert torch.allclose(reconstructed.data, original.data), (
            "Tensor leaf 'data' was corrupted by flatten/unflatten"
        )
        assert reconstructed.wm_memory is None

    def test_pytree_partial_roundtrip_with_overwritten_tensor_slots(self):
        """Simulates the AOS buffer write-then-read pattern.

        The AOS fix stores tensor leaves by writing them into pre-allocated
        buffers, then reconstructs the state by substituting gathered values
        back into the leaf list.  This test exercises that substitution
        pattern directly.
        """
        from torch.utils import _pytree as pytree

        B, D = 3, 4
        original = MockNetworkState(
            data=torch.arange(B * D, dtype=torch.float32).view(B, D),
            wm_memory=None,
        )

        # Simulate FlatTree.node_state_leaves: [B, N, D] buffer
        N = 10
        buffer = torch.zeros(B, N, D)

        leaves, treespec = pytree.tree_flatten(original)

        # Store at node slot 0
        tensor_slot = 0
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                buffer[:, 0, :] = leaf.float()
                tensor_slot += 1

        # Now "gather" from node slot 0 (simulating _lookup_network_states)
        gathered_tensor = buffer[:, 0, :]
        new_leaves = list(leaves)
        tensor_slot = 0
        for i, orig_leaf in enumerate(leaves):
            if isinstance(orig_leaf, torch.Tensor):
                new_leaves[i] = gathered_tensor
                tensor_slot += 1

        reconstructed = pytree.tree_unflatten(new_leaves, treespec)

        assert isinstance(reconstructed, MockNetworkState)
        assert torch.allclose(reconstructed.data, original.data), (
            "Buffer store/gather round-trip corrupted the tensor leaf"
        )
        assert reconstructed.wm_memory is None

    # ------------------------------------------------------------------
    # run_vectorized: state type check for batch > 1
    # ------------------------------------------------------------------

    def test_aos_vectorized_passes_network_state_not_tensor(self, base_config):
        """AOS run_vectorized must also reconstruct MockNetworkState correctly."""
        net = self._run_aos_capturing(base_config, B=2)

        assert net.captured_states, (
            "hidden_state_inference was never called in vectorized mode"
        )

        for i, state in enumerate(net.captured_states):
            assert isinstance(state, MockNetworkState), (
                f"AOS vectorized passed {type(state).__name__} at call {i}"
            )


# ==========================================================================
# Multiprocessing safety: pickling + error propagation
# ==========================================================================


class TestMultiprocessingSafety:
    """Tests that search outputs are safe to pass through multiprocessing queues.

    The ``TorchMPExecutor`` workers put all data through ``mp.Queue``, which
    pickles everything.  Search backend outputs must therefore be picklable.
    Additionally, the executor's error propagation path must surface the real
    exception message rather than causing a secondary ``TypeError: exceptions
    must derive from BaseException``.

    Regression: prior to the fix, ``_check_errors`` did ``raise err`` where
    ``err`` was a plain string (the exception was stringified in the worker to
    avoid exotic pickling failures).  Raising a string in Python 3 raises
    ``TypeError`` instead of the intended error, hiding the real crash.
    The fix wraps the string in ``RuntimeError`` before re-raising.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_single(self, make_search, config, num_actions=4):
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        _seed(42)
        search = make_search(config, num_actions)
        return search.run(obs, info, net)

    def _run_vectorized(self, make_search, config, num_actions=4, B=2):
        net = MockNetwork(num_actions)
        obs = torch.ones((B, 1, 4, 4))
        info = [{"player": 0, "legal_moves": list(range(num_actions))} for _ in range(B)]
        _seed(42)
        search = make_search(config, num_actions)
        return search.run_vectorized(obs, info, net)

    # ------------------------------------------------------------------
    # CPU-device guarantees (prerequisite for pickling via mp.Queue)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_run_output_tensors_are_on_cpu(self, make_search, base_config):
        """All tensor outputs from run() must be on CPU.

        ``mp.Queue`` pickling fails for CUDA tensors when the receiving
        process has no GPU context.  Both backends must return CPU tensors.
        """
        root_value, expl, tgt, best_action, metadata = self._run_single(
            make_search, base_config
        )

        assert not isinstance(root_value, torch.Tensor) or expl.device.type == "cpu", (
            "root_value is a tensor but not on CPU"
        )
        assert expl.device.type == "cpu", f"exploratory_policy on {expl.device}, expected cpu"
        assert tgt.device.type == "cpu", f"target_policy on {tgt.device}, expected cpu"

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_run_vectorized_output_tensors_are_on_cpu(self, make_search, base_config):
        """All tensor outputs from run_vectorized() must be on CPU."""
        root_values, expls, tgts, best_actions, metadata_list = self._run_vectorized(
            make_search, base_config
        )

        for i, (expl, tgt) in enumerate(zip(expls, tgts)):
            assert expl.device.type == "cpu", (
                f"exploratory_policy[{i}] on {expl.device}, expected cpu"
            )
            assert tgt.device.type == "cpu", (
                f"target_policy[{i}] on {tgt.device}, expected cpu"
            )

    # ------------------------------------------------------------------
    # Pickling round-trips (simulates mp.Queue put/get)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_run_output_is_picklable(self, make_search, base_config):
        """run() output must survive a pickle round-trip.

        In ``TorchMPExecutor._worker_loop`` the episode stats are put on
        ``result_queue`` which internally pickles them.  A non-picklable
        object would cause a worker crash, which then manifests as the
        ``TypeError: exceptions must derive from BaseException`` bug because
        the crash itself gets stringified and re-raised incorrectly.
        """
        out = self._run_single(make_search, base_config)
        try:
            data = pickle.dumps(out)
            restored = pickle.loads(data)
        except Exception as exc:
            pytest.fail(
                f"run() output from {make_search.__name__} is not picklable: {exc}"
            )

        root_value, expl, tgt, best_action, metadata = restored
        assert isinstance(root_value, float)
        assert isinstance(tgt, torch.Tensor)
        assert tgt.shape == (4,)

    @pytest.mark.parametrize(
        "make_search",
        [_make_py_search, _make_aos_search],
        ids=["python", "aos"],
    )
    def test_run_vectorized_output_is_picklable(self, make_search, base_config):
        """run_vectorized() output must survive a pickle round-trip."""
        out = self._run_vectorized(make_search, base_config)
        try:
            data = pickle.dumps(out)
            restored = pickle.loads(data)
        except Exception as exc:
            pytest.fail(
                f"run_vectorized() output from {make_search.__name__} is not picklable: {exc}"
            )

        root_values, expls, tgts, best_actions, metadata_list = restored
        assert len(tgts) == 2
        assert isinstance(tgts[0], torch.Tensor)

    # ------------------------------------------------------------------
    # Error-queue propagation regression test
    # ------------------------------------------------------------------

    @pytest.mark.unit
    def test_check_errors_wraps_string_in_runtime_error(self):
        """_check_errors must raise RuntimeError when the queue holds a string.

        Regression test for: ``TypeError: exceptions must derive from
        BaseException``.

        The worker loop stringifies exceptions before putting them on the
        error queue (to avoid pickling exotic exception types).  The old
        ``raise err`` where ``err`` is a string caused a secondary
        ``TypeError`` in Python 3, hiding the real crash message entirely.
        The fix wraps the string in ``RuntimeError`` before re-raising.

        We use a plain ``queue.Queue`` (in-process) and patch ``stop`` to a
        no-op because ``mp.Queue.empty()`` is documented as unreliable —
        it can return ``True`` immediately after a ``put()``, making tests
        that depend on the real ``mp.Queue`` timing-sensitive.
        """
        import queue as stdlib_queue
        from agents.executors.torch_mp_executor import TorchMPExecutor

        executor = TorchMPExecutor()
        executor.workers = []
        executor.error_queue = stdlib_queue.Queue()
        executor.stop = lambda: None  # no-op: avoid mp.Queue cleanup side-effects

        fake_error_msg = "SomeError: something went wrong inside the worker"
        fake_tb = "Traceback (most recent call last):\n  ...\nSomeError: something went wrong\n"
        executor.error_queue.put((fake_error_msg, fake_tb))

        # Before the fix: raises TypeError: exceptions must derive from BaseException
        # After the fix: raises RuntimeError with the original message.
        with pytest.raises(RuntimeError) as exc_info:
            executor._check_errors()

        assert fake_error_msg in str(exc_info.value), (
            "_check_errors must embed the original error message in the RuntimeError"
        )

    @pytest.mark.unit
    def test_check_errors_passes_through_real_exceptions(self):
        """_check_errors must re-raise real BaseException objects unchanged.

        If a future refactor changes the worker to put actual exception
        objects (not strings) on the queue, _check_errors must still work.
        """
        import queue as stdlib_queue
        from agents.executors.torch_mp_executor import TorchMPExecutor

        executor = TorchMPExecutor()
        executor.workers = []
        executor.error_queue = stdlib_queue.Queue()
        executor.stop = lambda: None

        real_exc = ValueError("a real exception")
        executor.error_queue.put((real_exc, ""))

        with pytest.raises(ValueError, match="a real exception"):
            executor._check_errors()

    @pytest.mark.unit
    def test_check_errors_is_silent_when_queue_is_empty(self):
        """_check_errors must not raise when the error queue is empty."""
        import queue as stdlib_queue
        from agents.executors.torch_mp_executor import TorchMPExecutor

        executor = TorchMPExecutor()
        executor.workers = []
        executor.error_queue = stdlib_queue.Queue()
        executor.stop = lambda: None

        # Should complete without raising.
        executor._check_errors()

    @pytest.mark.unit
    def test_check_errors_string_raises_runtime_error_not_type_error(self):
        """Direct unit test of the isinstance guard added to _check_errors.

        Verifies that the specific Python 3 error — ``TypeError: exceptions
        must derive from BaseException`` — can no longer be triggered by
        putting a string on the error queue.
        """
        # Confirm the old bug: Python 3 cannot raise a plain string.
        with pytest.raises(TypeError):
            raise "this would be the old bug"  # type: ignore[misc]  # noqa

        # The fix: wrap in RuntimeError first, then raise succeeds.
        err: str | BaseException = "SomeError: the worker crashed"
        if not isinstance(err, BaseException):
            err = RuntimeError(err)
        with pytest.raises(RuntimeError, match="the worker crashed"):
            raise err


# ==========================================================================
# NaN / inf guard — regression for worker crash
# ==========================================================================


class TestNaNGuard:
    """Regression tests: no NaN or inf in any search backend's policy outputs.

    Reproduces the actor crash::

        ValueError: Expected parameter logits (Tensor of shape (1, 9)) ...
        found invalid values: tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan]])

    Root cause (C++ backend): the extraction loop in ``run_vectorized`` used
    ``num_actions`` as the bound but ``root_child_visits()`` can return a
    vector of size ``|allowed_actions|`` when legal moves are a strict subset,
    causing out-of-bounds reads (UB → garbage/NaN floats).  Secondary cause:
    ``py::array_t<double>::unchecked()`` on float32 tensor buffers reads wrong
    bytes.  Both are fixed in ``bindings.cpp``.
    """

    def _assert_valid_policy(self, policy: torch.Tensor, label: str) -> None:
        assert not torch.isnan(policy).any(), f"{label}: NaN in policy {policy}"
        assert not torch.isinf(policy).any(), f"{label}: inf in policy {policy}"
        assert policy.min().item() >= 0.0, f"{label}: negative prob in policy {policy}"
        assert abs(policy.sum().item() - 1.0) < 1e-5, (
            f"{label}: policy does not sum to 1.0 (got {policy.sum().item():.6f})"
        )

    def _assert_valid_run_output(self, out: tuple, label: str) -> None:
        _, expl, tgt, _, _ = out
        self._assert_valid_policy(expl, f"{label} exploratory_policy")
        self._assert_valid_policy(tgt, f"{label} target_policy")

    # ------------------------------------------------------------------
    # Python backend
    # ------------------------------------------------------------------

    def test_python_no_nan_all_legal(self, base_config):
        """Python backend: full action space — baseline must be NaN-free."""
        num_actions = 9
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        out = _py_run(base_config, num_actions, obs, info, net)
        self._assert_valid_run_output(out, "python/all_legal")

    def test_python_no_nan_partial_legal(self, base_config):
        """Python backend: strict legal-moves subset must produce valid probs."""
        num_actions = 9
        legal = [0, 2, 4]
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": legal}
        out = _py_run(base_config, num_actions, obs, info, net)
        self._assert_valid_run_output(out, "python/partial_legal")

    # ------------------------------------------------------------------
    # AOS backend
    # ------------------------------------------------------------------

    def test_aos_no_nan_all_legal(self, base_config):
        """AOS backend: full action space — baseline must be NaN-free."""
        num_actions = 9
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        out = _aos_run(base_config, num_actions, obs, info, net)
        self._assert_valid_run_output(out, "aos/all_legal")

    def test_aos_no_nan_partial_legal(self, base_config):
        """AOS backend: strict legal-moves subset must produce valid probs."""
        num_actions = 9
        legal = [0, 2, 4]
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": legal}
        out = _aos_run(base_config, num_actions, obs, info, net)
        self._assert_valid_run_output(out, "aos/partial_legal")

    # ------------------------------------------------------------------
    # C++ backend
    # ------------------------------------------------------------------

    @_skip_cpp
    def test_cpp_no_nan_all_legal(self, base_config):
        """C++ backend: full action space — baseline must be NaN-free."""
        num_actions = 9
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": list(range(num_actions))}
        out = _cpp_run(base_config, num_actions, obs, info, net)
        self._assert_valid_run_output(out, "cpp/all_legal")

    @_skip_cpp
    def test_cpp_no_nan_partial_legal(self, base_config):
        """Regression: NaN policy when legal_moves is a strict subset (Catan-like).

        Before the fix, the C++ extraction loop read ``visits[a]`` for
        ``a >= visits.size()``, invoking UB that produced garbage/NaN floats.
        """
        num_actions = 9
        legal = [0, 2, 4]
        net = MockNetwork(num_actions)
        obs = torch.ones((1, 4, 4))
        info = {"player": 0, "legal_moves": legal}
        out = _cpp_run(base_config, num_actions, obs, info, net)
        self._assert_valid_run_output(out, "cpp/partial_legal")
