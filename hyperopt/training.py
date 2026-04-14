from .config import get_active_config
from .evaluation import test_score_evaluation, elo_evaluation, best_agent_elo_evaluation, test_agents_elo_evaluation

def _env_factory_safe(env_factory_fn):
    try: return env_factory_fn(render_mode="rgb_array")
    except TypeError: return env_factory_fn()

def marl_run_training(params, agent_name):
    config = get_active_config()
    params = config.prep_params(params)
    env = _env_factory_safe(config.env_factory)
    agent = config.agent_class(env=env, config=config.agent_config(config_dict=params, game_config=config.game_config(env_factory=config.env_factory)), name=agent_name, device=config.device, test_agents=config.test_agents)
    agent.checkpoint_interval, agent.test_interval, agent.test_trials = config.checkpoint_interval, config.test_interval, config.test_trials
    agent.train()
    if config.eval_method == "elo": return elo_evaluation(agent)
    elif config.eval_method == "best_agent_elo": return best_agent_elo_evaluation(agent)
    elif config.eval_method == "test_agents_elo": return test_agents_elo_evaluation(agent)
    raise NotImplementedError(f"Unknown eval method: {config.eval_method}")

def sarl_run_training(params, agent_name):
    config = get_active_config()
    params = config.prep_params(params)
    env = _env_factory_safe(config.env_factory)
    agent = config.agent_class(env=env, config=config.agent_config(config_dict=params, game_config=config.game_config(env_factory=config.env_factory)), name=agent_name, device=config.device)
    agent.checkpoint_interval, agent.test_interval, agent.test_trials = config.checkpoint_interval, config.test_interval, config.test_trials
    agent.train()
    return test_score_evaluation(agent, eval_method=config.eval_method, num_trials=config.test_trials, last_n=config.last_n_rolling_avg)
