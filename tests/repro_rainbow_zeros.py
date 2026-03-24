import torch
from configs.games.cartpole import CartPoleConfig
from configs.agents.rainbow_dqn import RainbowConfig
from modules.models.agent_network import AgentNetwork
from agents.learner.losses.representations import get_representation

def repro():
    # 1. Setup Game Config
    game_config = CartPoleConfig()
    game_config.min_score = 0.0
    game_config.max_score = 200.0
    print(f"Game Config: min={game_config.min_score}, max={game_config.max_score}")

    # 2. Setup Rainbow Config
    config_dict = {
        "agent_type": "rainbow",
        "action_selector": {"base": {"type": "argmax"}},
        "atom_size": 51,
        "dueling": True,
        "backbone": {"type": "mlp", "widths": [128]},
        "head": {
            "output_strategy": {
                "type": "c51",
                "num_atoms": 51,
                "v_min": 0.0,
                "v_max": 200.0,
            },
            "value_hidden_backbone": {"type": "mlp", "widths": [128]},
            "advantage_hidden_backbone": {"type": "mlp", "widths": [128]},
        }
    }
    config = RainbowConfig(config_dict, game_config)
    
    # 3. Create Network
    input_shape = (4,)
    num_actions = 2
    
    # Extract heads config from RainbowConfig
    heads_config = {
        "q_logits": config.head
    }
    
    model = AgentNetwork(
        input_shape=input_shape,
        num_actions=num_actions,
        arch_config=config.arch,
        heads_config=heads_config,
    )
    model.initialize()
    
    # 4. Check Representation and Support
    head = model.components["behavior_heads"]["q_logits"]
    rep = head.representation
    print(f"Representation type: {type(rep)}")
    print(f"Support range: {rep.vmin} to {rep.vmax}")
    print(f"Support shape: {rep.support.shape}")
    print(f"Support mean: {rep.support.mean().item()}")
    
    # 5. Dummy Forward Pass
    obs = torch.randn(1, 4)
    out = model.obs_inference(obs)
    
    print(f"Q-values shape: {out.q_values.shape}")
    print(f"Q-values (Expected Value): {out.q_values}")
    print(f"Q-values Mean: {out.q_values.mean().item()}")
    
    # 6. Check Metrics
    telemetry = out.extras
    print(f"Metrics (telemetry): {telemetry}")
    print(f"Q-Mean from telemetry: {telemetry.get('q_logits/mean')}")

if __name__ == "__main__":
    repro()
