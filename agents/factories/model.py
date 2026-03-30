import torch
from typing import Tuple, Any, Dict, Optional, Callable
from torch import nn
from functools import partial

from modules.models.agent_network import AgentNetwork
from modules.models.world_model import WorldModel
from agents.factories.builders import make_backbone_fn, make_head_fn

def build_agent_network(
    config: Any,
    obs_dim: Tuple[int, ...],
    num_actions: int,
    device: torch.device = torch.device("cpu"),
) -> AgentNetwork:
    """
    Assembles a modular AgentNetwork from a high-level configuration object.
    
    This factory orchestrates the creation of representation backbones, world models,
    and behavior heads using specialized component builders. It ensures that 
    environmental components (dynamics, encoder) are correctly encapsulated 
    and behavior components (policy, value) are properly routed.
    
    Args:
        config: The configuration object (e.g., MuZeroConfig).
        obs_dim: The observation shape (C, H, W) or (D,).
        num_actions: The number of actions in the environment.
        device: The device to place the network on.
        
    Returns:
        A fully built and initialized AgentNetwork.
    """
    # 1. Build Representation Backbone
    # representation_backbone is the entry point for all observations.
    representation_fn = make_backbone_fn(getattr(config, "representation_backbone", None))
    
    # 2. Build World Model Assembly (Environment Phase)
    world_model_fn = None
    # Many agents (like MuZero) encapsulate their world model components in a sub-config.
    # If config itself is the WorldModel config (as in MuZeroConfig), we use it.
    wm_cfg = getattr(config, "world_model", None)
    if wm_cfg is not None:
        # Build Environment Heads (Reward, ToPlay, Continuation, etc.)
        env_head_fns = {}
        cfg_env_heads = getattr(wm_cfg, "env_heads", {})
        for name, head_cfg in cfg_env_heads.items():
            env_head_fns[name] = make_head_fn(
                head_cfg,
                num_players=config.game.num_players,
                num_actions=num_actions,
            )
            
        # Build Dynamics Backbone
        dynamics_fn = make_backbone_fn(getattr(wm_cfg, "dynamics_backbone", None))
        
        # Build Optional Stochastic/Afterstate Components
        stochastic = getattr(wm_cfg, "stochastic", False)
        afterstate_dynamics_fn = make_backbone_fn(getattr(wm_cfg, "afterstate_dynamics_backbone", None))
        
        # sigma_head_fn (for chance prediction in stochastic MuZero)
        # In current config architecture, it's called 'chance_probability_head'
        sigma_head_config = getattr(wm_cfg, "chance_probability_head", None)
        sigma_head_fn = make_head_fn(sigma_head_config)
        
        # chance_encoder_fn (for encoding next obs into chance codes)
        chance_encoder_fn = make_backbone_fn(getattr(wm_cfg, "chance_encoder_backbone", None))
        
        # Define the world model builder closure
        def wm_builder(latent_dimensions: Tuple[int, ...], num_actions: int, num_players: int) -> WorldModel:
            return WorldModel(
                latent_dimensions=latent_dimensions,
                num_actions=num_actions,
                num_players=num_players,
                stochastic=stochastic,
                num_chance=getattr(wm_cfg, "num_chance", 0),
                observation_shape=obs_dim,
                use_true_chance_codes=getattr(wm_cfg, "use_true_chance_codes", False),
                env_head_fns=env_head_fns,
                dynamics_fn=dynamics_fn,
                afterstate_dynamics_fn=afterstate_dynamics_fn,
                sigma_head_fn=sigma_head_fn,
                encoder_fn=chance_encoder_fn,
                action_embedding_dim=getattr(wm_cfg, "action_embedding_dim", 32),
                is_discrete=config.game.is_discrete,
                # is_spatial is inferred by WorldModel/get_action_encoder from latent_dimensions
            )
        world_model_fn = wm_builder

    # 3. Build Behavior Heads (Policy, Value, Q, etc.)
    head_fns = {}
    cfg_behavior_heads = getattr(config, "heads", {})
    for name, head_cfg in cfg_behavior_heads.items():
        head_fns[name] = make_head_fn(
            head_cfg,
            num_players=config.game.num_players,
            num_actions=num_actions,
        )
        
    # 4. Build Temporal Backbone (Memory Core like LSTM/Transformer for AgentNetwork levels)
    memory_core_fn = make_backbone_fn(getattr(config, "memory_core", None))
    
    # 5. Assemble AgentNetwork
    network = AgentNetwork(
        input_shape=obs_dim,
        num_actions=num_actions,
        num_players=config.game.num_players,
        representation_fn=representation_fn,
        world_model_fn=world_model_fn,
        memory_core_fn=memory_core_fn,
        head_fns=head_fns,
        stochastic=getattr(config, "stochastic", False),
        num_chance_codes=getattr(config, "num_chance", 0),
    )
    
    return network.to(device)
