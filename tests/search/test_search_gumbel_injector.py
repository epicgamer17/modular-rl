import pytest
pytestmark = pytest.mark.unit

from types import SimpleNamespace

import torch
from torch.distributions import Categorical

from search.initial_searchsets import SelectTopK
from search.prior_injectors import GumbelInjector


def test_gumbel_injector_keeps_illegal_moves_at_zero_probability():
    torch.manual_seed(0)
    legal_moves = [0, 2]

    # Illegal moves are explicitly masked in logits.
    logits = torch.tensor([1.5, -float("inf"), 0.5, -float("inf")])
    dist = Categorical(logits=logits)
    policy = dist.probs

    injector = GumbelInjector()
    noisy_policy = injector.inject(
        policy=policy,
        legal_moves=legal_moves,
        config=SimpleNamespace(),
        policy_dist=dist,
    )

    illegal = torch.tensor([1, 3])
    legal = torch.tensor(legal_moves)

    assert torch.all(noisy_policy[illegal] == 0.0)
    assert torch.isclose(noisy_policy[legal].sum(), torch.tensor(1.0))


def test_select_topk_only_selects_legal_actions():
    priors = torch.tensor([0.95, 0.05, 0.90, 0.80])
    legal_moves = [1, 3]

    selected = SelectTopK().create_initial_searchset(
        priors=priors,
        legal_moves=legal_moves,
        count=2,
    )

    assert set(selected.tolist()).issubset(set(legal_moves))
