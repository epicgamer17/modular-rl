import argparse
import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Type


class RobustLoader(yaml.Loader):
    def construct_python_object(self, suffix, node):
        return f"UnknownObject:{suffix}"

    def construct_python_name(self, suffix, node):
        return f"UnknownName:{suffix}"

    def construct_undefined(self, node):
        return None


RobustLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:", RobustLoader.construct_python_object
)
RobustLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", RobustLoader.construct_python_name
)
RobustLoader.add_multi_constructor(
    "!!python/object:", RobustLoader.construct_python_object
)
RobustLoader.add_multi_constructor("!!python/name:", RobustLoader.construct_python_name)
RobustLoader.add_constructor(None, RobustLoader.construct_undefined)

# Monkeypatch yaml.Loader used by AgentConfig.load
yaml.Loader = RobustLoader

# Import repo components
try:
    from configs.base import ConfigBase
    from configs.games.game import GameConfig
    from configs.agents.base import AgentConfig

    # 1. Monkeypatch ConfigBase.parse_field
    orig_parse_field = ConfigBase.parse_field

    def robust_parse_field(
        self, field_name, default=None, wrapper=None, required=True, dtype=None
    ):
        if field_name not in self.config_dict and default is None and required:
            return None
        return orig_parse_field(self, field_name, default, wrapper, required, dtype)

    ConfigBase.parse_field = robust_parse_field

    # 2. Monkeypatch AgentConfig.__init__ and _verify_game
    orig_agent_config_init = AgentConfig.__init__

    def robust_agent_config_init(self, config_dict, game_config):
        if (
            game_config is None
            or not hasattr(game_config, "num_actions")
            or "Unknown" in str(game_config)
        ):
            print("Note: Providing dummy GameConfig during robust initialization.")
            game_config = GameConfig(
                max_score=500,
                min_score=0,
                is_discrete=True,
                is_image=False,
                is_deterministic=True,
                has_legal_moves=False,
                perfect_information=True,
                multi_agent=False,
                num_players=1,
                num_actions=2,
                make_env=None,
            )
        orig_agent_config_init(self, config_dict, game_config)

    def robust_verify_game(self):
        pass

    AgentConfig.__init__ = robust_agent_config_init
    AgentConfig._verify_game = robust_verify_game

    from modules.models.agent_network import AgentNetwork
    from configs.agents.ppo import PPOConfig
    from configs.agents.muzero import MuZeroConfig
    from configs.agents.rainbow_dqn import RainbowConfig
except ImportError as e:
    print(
        f"Error: Could not import repository modules. Ensure this script is run from the project root or its child directories. {e}"
    )
    sys.exit(1)

AGENT_MAPPING = {
    "ppo": {"config": PPOConfig, "network": AgentNetwork},
    "muzero": {"config": MuZeroConfig, "network": AgentNetwork},
    "dqn": {"config": RainbowConfig, "network": AgentNetwork},
    "rainbow": {"config": RainbowConfig, "network": AgentNetwork},
}


def get_input_shape(config, env=None) -> Tuple[int, ...]:
    if env is not None:
        try:
            obs_space = env.observation_space
            if hasattr(obs_space, "shape"):
                return tuple(obs_space.shape)
            elif callable(obs_space):
                try:
                    space = obs_space("player_0")
                    return tuple(space.shape)
                except:
                    pass
        except:
            pass
    return (4,)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a PyTorch model in the rl-research repository."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to a checkpoint folder (contains configs/config.yaml).",
    )
    parser.add_argument(
        "--weights_path", type=str, help="Direct path to .pt or .pth weights file."
    )
    parser.add_argument("--config_path", type=str, help="Direct path to config.yaml.")
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=AGENT_MAPPING.keys(),
        help="Agent type for instantiation.",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        help="Explicit input shape excluding batch (e.g. 3 224 224).",
    )
    parser.add_argument(
        "--output_path", type=str, help="Custom output path for the PNG."
    )
    args = parser.parse_args()

    try:
        from torchviz import make_dot
    except ImportError:
        print("Error: 'torchviz' not found. Please install it: pip install torchviz")
        sys.exit(1)

    config = None
    weights = None
    agent_type = args.agent_type

    if args.checkpoint_dir:
        cp_path = Path(args.checkpoint_dir)
        config_path = cp_path / "configs/config.yaml"
        weights_path = cp_path / "model_weights/weights.pt"
        if not weights_path.exists():
            step_dirs = list(cp_path.glob("step_*"))
            if step_dirs:
                try:
                    latest_step = sorted(
                        step_dirs,
                        key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0,
                    )[-1]
                    weights_path = latest_step / "model_weights/weights.pt"
                except (IndexError, ValueError):
                    pass
        if config_path.exists() and not args.config_path:
            args.config_path = str(config_path)
        if weights_path.exists() and not args.weights_path:
            args.weights_path = str(weights_path)
        if not agent_type:
            name_lower = str(cp_path).lower()
            if "ppo" in name_lower:
                agent_type = "ppo"
            elif "muzero" in name_lower:
                agent_type = "muzero"
            elif "dqn" in name_lower or "rainbow" in name_lower:
                agent_type = "rainbow"

    if args.config_path and agent_type:
        print(f"Loading {agent_type} config from {args.config_path}...")
        try:
            config_cls = AGENT_MAPPING[agent_type]["config"]
            config = config_cls.load(args.config_path)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    if args.weights_path:
        print(f"Loading weights from {args.weights_path}...")
        try:
            weights = torch.load(
                args.weights_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)

    model = None
    inference_shape = None

    if agent_type and config:
        print(f"Instantiating {agent_type} network...")
        try:
            net_cls = AGENT_MAPPING[agent_type]["network"]
            env = None
            if (
                hasattr(config, "game")
                and hasattr(config.game, "make_env")
                and config.game.make_env
            ):
                try:
                    if "UnknownObject" not in str(config.game.make_env):
                        env = config.game.make_env()
                except Exception as e:
                    print(f"Note: Could not create environment to infer shapes: {e}")
            inference_shape = (
                tuple(args.input_shape)
                if args.input_shape
                else get_input_shape(config, env)
            )
            num_actions = getattr(config.game, "num_actions", 2)
            print(
                f"Resolved input shape: {inference_shape}, num_actions: {num_actions}"
            )

            model = net_cls(
                config=config,
                input_shape=inference_shape,
                num_actions=num_actions,
            )

            if weights:
                state_dict = (
                    weights.get("model", weights)
                    if isinstance(weights, dict)
                    else weights
                )
                if isinstance(state_dict, dict):
                    print("Loading state dict into instantiated model...")
                    model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error during model instantiation: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    elif weights and isinstance(weights, torch.nn.Module):
        model = weights
        print("Loaded full model object from file.")
    elif weights and isinstance(weights, dict):
        print(
            "File contains only weights (state_dict). Provide --agent_type and --config_path."
        )
        sys.exit(1)
    else:
        print("Error: Could not determine how to build or load the model.")
        sys.exit(1)

    print("Generating graph...")
    model.eval()

    if inference_shape:
        dummy_input = torch.randn(1, *inference_shape)
    elif args.input_shape:
        dummy_input = torch.randn(1, *args.input_shape)
    else:
        dummy_input = (
            torch.randn(1, 4) if agent_type != "muzero" else torch.randn(1, 3, 64, 64)
        )

    try:
        with torch.no_grad():
            output = model(dummy_input)

        if isinstance(output, tuple):
            print(f"Model returned {len(output)} outputs. Visualizing the first one.")
            output = output[0]

        dot = make_dot(output, params=dict(model.named_parameters()))

        if args.output_path:
            final_output = args.output_path
        elif args.checkpoint_dir:
            final_output = str(Path(args.checkpoint_dir) / f"{agent_type}_graph")
        elif args.weights_path:
            final_output = str(Path(args.weights_path).with_suffix("")) + "_graph"
        else:
            final_output = f"{agent_type or 'model'}_graph"

        dot.format = "png"
        try:
            dot.render(final_output)
            print(f"Success! Architecture graph saved to {final_output}.png")
        except Exception as e:
            if "ExecutableNotFound" in str(type(e)) or "dot" in str(e):
                print(
                    f"Warning: Graphviz 'dot' executable not found. Saving DOT source to {final_output}.gv instead."
                )
                print(
                    "To view it, install Graphviz (brew install graphviz or apt-get install graphviz) or paste the content into https://dreampuf.github.io/GraphvizOnline/"
                )
                dot.save(f"{final_output}.gv")
                print(f"DOT source saved to {final_output}.gv")
            else:
                raise e

    except Exception as e:
        print(f"Error during visualization: {e}")
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
