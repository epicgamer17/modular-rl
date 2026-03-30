import pytest
import search.nodes as search_nodes
import torch
from torch.distributions import Categorical

from agents.environments.adapters import PettingZooAdapter
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from modules.models.inference_output import InferenceOutput
from stats.stats import StatTracker

pytestmark = pytest.mark.integration


def _make_trainer(make_muzero_config_dict, tictactoe_game_config):
    cfg_dict = make_muzero_config_dict(
        num_workers=0,
        multi_process=False,
        search_backend="python",
        num_simulations=1,
        search_batch_size=1,
        use_dirichlet=False,
        to_play_head={"output_strategy": {"type": "categorical"}},
        known_bounds=[tictactoe_game_config.min_score, tictactoe_game_config.max_score],
    )
    config = MuZeroConfig(cfg_dict, tictactoe_game_config)
    env = tictactoe_game_config.env_factory()
    return MuZeroTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="test_tictactoe_search_to_play",
        stats=StatTracker("test_tictactoe_search_to_play"),
    )


class DummySearchNetwork(torch.nn.Module):
    def __init__(self, num_actions: int, recurrent_to_play: int):
        super().__init__()
        self.num_actions = num_actions
        self.recurrent_to_play = recurrent_to_play

    def _policy(self, batch_size: int) -> Categorical:
        logits = torch.full((batch_size, self.num_actions), -5.0)
        logits[:, 0] = 5.0
        return Categorical(logits=logits)

    def obs_inference(self, obs):
        batch_size = obs.shape[0]
        return InferenceOutput(
            value=torch.zeros(batch_size, dtype=torch.float32),
            policy=self._policy(batch_size),
            recurrent_state={"latent": torch.zeros(batch_size, 1)},
        )

    def hidden_state_inference(self, recurrent_state, action):
        batch_size = action.shape[0]
        return InferenceOutput(
            value=torch.zeros(batch_size, dtype=torch.float32),
            reward=torch.zeros(batch_size, dtype=torch.float32),
            policy=self._policy(batch_size),
            recurrent_state={"latent": torch.ones(batch_size, 1)},
            to_play=torch.full(
                (batch_size,), self.recurrent_to_play, dtype=torch.long
            ),
        )


def test_muzero_trainer_search_policy_source_forwards_to_play_kwarg(
    make_muzero_config_dict, tictactoe_game_config, monkeypatch
):
    """
    Tier 2 integration test:
    MuZeroTrainer's SearchPolicySource must forward the actor's explicit `to_play`
    into search info when the environment payload does not already contain `player`.
    """
    trainer = _make_trainer(make_muzero_config_dict, tictactoe_game_config)
    adapter = PettingZooAdapter(
        tictactoe_game_config.env_factory,
        device=torch.device("cpu"),
        num_actions=tictactoe_game_config.num_actions,
    )
    obs, info = adapter.reset()

    assert int(info["player_id"][0].item()) == 0
    info = {k: v for k, v in info.items() if k != "player"}
    assert "player" not in info

    seen = {}

    def fake_run_vectorized(obs_arg, info_arg, agent_network_arg):
        seen["player"] = info_arg["player"]
        uniform = torch.full(
            (tictactoe_game_config.num_actions,),
            1.0 / tictactoe_game_config.num_actions,
            dtype=torch.float32,
        )
        return ([0.0], [uniform], [uniform], [0], [{}])

    monkeypatch.setattr(
        trainer.search_policy_source.search, "run_vectorized", fake_run_vectorized
    )

    trainer.search_policy_source.get_inference(
        obs,
        info,
        agent_network=DummySearchNetwork(tictactoe_game_config.num_actions, 1),
        to_play=1,
    )

    assert seen["player"] == 1


def test_muzero_tictactoe_search_uses_recurrent_to_play_for_child_expansion(
    make_muzero_config_dict, tictactoe_game_config, monkeypatch
):
    """
    Tier 2 integration test:
    MuZero Tic-Tac-Toe search must expand the root with the environment player
    and expand the next decision node with the world-model-predicted `to_play`.
    """
    trainer = _make_trainer(make_muzero_config_dict, tictactoe_game_config)
    adapter = PettingZooAdapter(
        tictactoe_game_config.env_factory,
        device=torch.device("cpu"),
        num_actions=tictactoe_game_config.num_actions,
    )
    obs, info = adapter.reset()

    root_player = int(info["player_id"][0].item())
    info = {**info, "player": root_player}

    recorded_expansions = []
    original_expand = search_nodes.DecisionNode.expand

    def recording_expand(
        self,
        allowed_actions,
        to_play,
        priors,
        network_policy,
        network_state,
        reward,
        value=None,
        network_policy_dist=None,
    ):
        label = "root" if self.parent is None else "child"
        recorded_expansions.append((label, int(to_play)))
        return original_expand(
            self,
            allowed_actions=allowed_actions,
            to_play=to_play,
            priors=priors,
            network_policy=network_policy,
            network_state=network_state,
            reward=reward,
            value=value,
            network_policy_dist=network_policy_dist,
        )

    monkeypatch.setattr(search_nodes.DecisionNode, "expand", recording_expand)

    trainer.search_policy_source.search.run_vectorized(
        obs,
        info,
        DummySearchNetwork(tictactoe_game_config.num_actions, recurrent_to_play=1),
    )

    assert len(recorded_expansions) >= 2
    assert recorded_expansions[0] == ("root", root_player)
    assert recorded_expansions[1] == ("child", 1)
