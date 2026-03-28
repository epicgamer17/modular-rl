import torch
from modules.models.agent_network import AgentNetwork
from modules.models.world_model import WorldModel
from agents.factories.builders import make_backbone_fn, make_head_fn
from functools import partial

def test_stochastic_path():
    print("Testing Stochastic Path Architecture...")
    
    # Mock config-like structure
    class MockConfig:
        def __init__(self):
            self.stochastic = True
            self.num_chance = 4
            self.unroll_steps = 5
            self.minibatch_size = 8
            self.representation_backbone = {"type": "mlp", "h_dim": 16, "num_layers": 1}
            self.prediction_backbone = {"type": "mlp", "h_dim": 16, "num_layers": 1}
            self.world_model = type('WM', (), {
                'stochastic': True,
                'num_chance': 4,
                'game': type('G', (), {'observation_shape': (10,)})(),
                'use_true_chance_codes': False,
                'env_heads': {'reward': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1}},
                'dynamics_backbone': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1},
                'afterstate_dynamics_backbone': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1},
                'chance_probability_head': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1},
                'chance_encoder_backbone': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1},
                'action_embedding_dim': 8
            })()
            self.heads = {
                'policy': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1},
                'value': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1},
                'afterstate_value': {'type': 'mlp', 'h_dim': 16, 'num_layers': 1}
            }
            self.game = type('G', (), {'num_players': 1, 'num_actions': 4})()

    config = MockConfig()
    device = torch.device("cpu")
    
    # Build components (mimicking MuZeroTrainer)
    representation_fn = make_backbone_fn(config.representation_backbone)
    prediction_backbone_fn = make_backbone_fn(config.prediction_backbone)
    
    wm_cfg = config.world_model
    env_head_fns = {name: make_head_fn(h_cfg) for name, h_cfg in wm_cfg.env_heads.items()}
    
    world_model_fn = partial(
        WorldModel,
        stochastic=wm_cfg.stochastic,
        num_chance=wm_cfg.num_chance,
        observation_shape=wm_cfg.game.observation_shape,
        use_true_chance_codes=wm_cfg.use_true_chance_codes,
        env_head_fns=env_head_fns,
        dynamics_fn=make_backbone_fn(wm_cfg.dynamics_backbone),
        afterstate_dynamics_fn=make_backbone_fn(wm_cfg.afterstate_dynamics_backbone),
        sigma_head_fn=make_head_fn(wm_cfg.chance_probability_head),
        encoder_fn=make_backbone_fn(wm_cfg.chance_encoder_backbone),
        action_embedding_dim=wm_cfg.action_embedding_dim,
    )
    
    head_fns = {name: make_head_fn(h_cfg) for name, h_cfg in config.heads.items()}
    
    agent_network = AgentNetwork(
        input_shape=(10,),
        num_actions=4,
        representation_fn=representation_fn,
        world_model_fn=world_model_fn,
        prediction_backbone_fn=prediction_backbone_fn,
        head_fns=head_fns,
        stochastic=config.stochastic,
        num_players=1,
        num_chance_codes=4,
    ).to(device)

    # 1. Verify backbone injection
    wm = agent_network.components["world_model"]
    assert hasattr(wm.dynamics_pipeline, "prediction_backbone"), "StochasticDynamics missing prediction_backbone"
    assert wm.dynamics_pipeline.prediction_backbone is agent_network.components["prediction_backbone"], "Backbone not injected correctly into WorldModel"
    print("✓ Backbone injection verified.")

    # 2. Verify afterstate_inference guard and execution
    obs = torch.randn(1, 10)
    out_obs = agent_network.obs_inference(obs)
    recurrent_state = out_obs.recurrent_state
    action = torch.zeros((1,), dtype=torch.long)
    
    # This should NOT raise NotImplementedError now
    out_as = agent_network.afterstate_inference(recurrent_state, action)
    assert out_as.afterstate_features is not None, "Afterstate features should be present"
    assert "chance" in out_as.extras, "Chance logits should be present in extras"
    print("✓ afterstate_inference guard and execution verified.")

    print("Stochastic Path verification PASSED.")

if __name__ == "__main__":
    test_stochastic_path()
