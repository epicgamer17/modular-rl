import numpy as np
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import Dict, Optional, List, Any
import matplotlib.pyplot as plt
from enum import Enum, auto
from pathlib import Path
from stats.latent_pca import LatentPCAVisualizer
from stats.latent_tsne import LatentTSNEVisualizer

try:
    from stats.latent_umap import LatentUMAPVisualizer
except ImportError:
    LatentUMAPVisualizer = None


class PlotType(Enum):
    ROLLING_AVG = auto()
    VARIATION_FILL = auto()
    BEST_FIT_LINE = auto()
    LOG_Y = auto()
    EXPONENTIAL_AVG = auto()
    BAR = auto()


class StatTracker:
    """
    A stat tracker that encapsulates multiprocessing queue logic.
    Can be instantiated as a 'host' (manages data and a queue)
    or a 'client' (sends data to the host's queue).
    """

    def __init__(
        self,
        name: str,
        stat_keys: List[str] = None,
        target_values: Optional[Dict[str, float]] = None,
        use_tensor_dicts: Optional[Dict[str, List[str]]] = None,
        # This new argument enables the dual-mode logic
        queue: Optional[mp.Queue] = None,
    ):
        self.name = name
        self._is_client = queue is not None

        if self._is_client:
            # CLIENT MODE: Only holds a reference to the queue. No data storage.
            self.queue = queue
            self.stats = None  # Clients don't store stats.
        else:
            # HOST MODE: Manages the queue and the actual data.
            self.queue = mp.Queue()
            self.stats = {}
            self.num_steps = 0
            self.time_elapsed = 0.0
            self.targets = target_values or {}
            use_tensor_dicts = use_tensor_dicts or {}
            self.plot_configs = {}
            self.latent_viz_data = {}
            self.custom_viz_data = {}

            if stat_keys:
                for key in stat_keys:
                    self._init_key(
                        key,
                        target_value=self.targets.get(key),
                        subkeys=use_tensor_dicts.get(key),
                    )

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure new attributes exist even in legacy objects
        if not hasattr(self, "custom_viz_data"):
            self.custom_viz_data = {}
        if not hasattr(self, "latent_viz_data"):
            self.latent_viz_data = {}
        if not hasattr(self, "plot_configs"):
            self.plot_configs = {}
        if not hasattr(self, "targets"):
            self.targets = {}
        if not hasattr(self, "time_elapsed"):
            self.time_elapsed = 0.0
        if not hasattr(self, "num_steps"):
            self.num_steps = 0

    def get_client(self) -> "StatTracker":
        """Returns a lightweight client instance for passing to a worker process."""
        if self._is_client:
            raise RuntimeError("Cannot get a client from another client.")
        # The client is initialized with the host's queue and no data keys.
        return StatTracker(name=self.name, queue=self.queue)

    def _init_key(
        self,
        key: str,
        target_value: Optional[float] = None,
        subkeys: Optional[List[str]] = None,
    ):
        """Internal method for the host to initialize a new stat."""
        print(f"Initializing stat '{key}' with subkeys {subkeys}")
        if key in self.stats:
            raise ValueError(f"Stat '{key}' already exists")
        if subkeys:
            self.stats[key] = {subkey: [] for subkey in subkeys}
        else:
            self.stats[key] = []
        if target_value is not None:
            self.targets[key] = target_value

    def append(self, key: str, value: float, subkey: Optional[str] = None):
        if self._is_client:
            # print("Appending stat {} from client".format(key))
            # Client sends the command to the queue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.queue.put(("append", key, value, subkey))
        else:
            # Handle dictionary values by recursively appending with subkeys
            if isinstance(value, dict):
                for k, v in value.items():
                    # If a subkey was already provided, we could nesting or just use the dict key
                    # Given the current usage, if it's a dict, the keys are usually the intended subkeys
                    self.append(key, v, subkey=k)
                return

            # Host executes the command directly
            if key not in self.stats:
                if subkey is not None:
                    self._init_key(key, subkeys=[subkey])
                else:
                    self._init_key(key)

            # Ensure values are detached and moved to CPU if they are tensors
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    new_val = float(value.detach().cpu().item())
                else:
                    new_val = value.detach().cpu()
            else:
                # Cast numpy scalars or basic numbers to standard python floats for cleaner memory
                new_val = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else value
                )

            if isinstance(self.stats[key], Dict):
                if subkey is None:
                    raise ValueError(f"Stat '{key}' requires a subkey")
                if subkey not in self.stats[key]:
                    self.stats[key][subkey] = []
                self.stats[key][subkey].append(new_val)
            else:
                self.stats[key].append(new_val)

    def set(self, key: str, value: Any, subkey: Optional[str] = None):
        """Sets a stat to a single value, replacing any existing history."""
        if self._is_client:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.queue.put(("set", key, value, subkey))
        else:
            if isinstance(value, dict):
                for k, v in value.items():
                    self.set(key, v, subkey=k)
                return

            if key not in self.stats:
                if subkey is not None:
                    self._init_key(key, subkeys=[subkey])
                else:
                    self._init_key(key)

            # Ensure values are detached and moved to CPU if they are tensors
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    new_val = float(value.detach().cpu().item())
                else:
                    new_val = value.detach().cpu()
            else:
                new_val = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else value
                )

            if isinstance(self.stats[key], Dict):
                if subkey is None:
                    raise ValueError(f"Stat '{key}' requires a subkey")
                self.stats[key][subkey] = [new_val]
            else:
                self.stats[key] = [new_val]

    def increment_steps(self, n: int = 1):
        if self._is_client:
            self.queue.put(("increment_steps", n))
        else:
            self.num_steps += n

    def drain_queue(self):
        """Host-only method to process all messages from clients."""
        if self._is_client:
            raise RuntimeError(
                "drain_queue() can only be called on the host StatTracker."
            )
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                method_name, *args = message
                getattr(self, method_name)(*args)
            except Empty:
                break
            except Exception as e:
                print(f"Error processing stat queue message: {message}, Error: {e}")

    # Other methods (get, plot_graphs, etc.) are host-only and don't need changes.
    # They will correctly fail on a client because `self.stats` is None.
    def get_num_steps(self):
        if self._is_client:
            return None
        return self.num_steps

    def set_time_elapsed(self, time_elapsed: float):
        if self._is_client:
            self.queue.put(("set_time_elapsed", time_elapsed))
        else:
            self.time_elapsed = time_elapsed

    # Other methods like plot_graphs, get, keys, etc. remain unchanged...
    def get_time_elapsed(self):
        if self._is_client:
            return None
        return self.time_elapsed

    def add_plot_types(self, key: str, *plot_types: PlotType, **params: Any):
        """Add extra plot types on top of current ones."""
        if self._is_client:
            return None
        if key not in self.plot_configs:
            self.plot_configs[key] = {"types": set(), "params": {}}
        self.plot_configs[key]["types"].update(plot_types)
        self.plot_configs[key]["params"].update(params)

    def add_latent_visualization(
        self,
        key: str,
        latents: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        method: str = "pca",
        **kwargs,
    ):
        """
        Updates the latent visualization data for a given key.
        Only the latest batch is kept.
        """
        # Detach and move to CPU immediately to avoid holding GPU memory in buffer
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu()

        if self._is_client:
            self.queue.put(
                ("add_latent_visualization", key, latents, labels, method, kwargs)
            )
        else:
            self.latent_viz_data[key] = {
                "latents": latents,
                "labels": labels,
                "method": method,
                "kwargs": kwargs,
            }

    def add_custom_visualization(
        self, key: str, data: Any, visualizer: Any = None, **kwargs
    ):
        """
        Updates custom visualization data for a given key.
        """
        if self._is_client:
            self.queue.put(("add_custom_visualization", key, data, visualizer, kwargs))
        else:
            self.custom_viz_data[key] = {
                "data": data,
                "visualizer": visualizer,
                "kwargs": kwargs,
            }

    def plot_graphs(self, dir: Optional[str] = None):
        if self._is_client:
            raise RuntimeError("Cannot plot graphs from a client instance.")

        collected_stats = {}
        for key, data in self.stats.items():
            if isinstance(data, Dict):
                if any(len(v) > 0 for v in data.values()):
                    collected_stats[key] = data
            elif len(data) > 0:
                collected_stats[key] = data

        fig = None
        if len(collected_stats) > 0:
            fig, axs = plt.subplots(
                len(collected_stats), 1, figsize=(10, 5 * len(collected_stats))
            )
            if len(collected_stats) == 1:
                axs = [axs]

            # Group subkeys by base key to handle consolidated plotting
            for ax, key in zip(axs, collected_stats.keys()):
                data = collected_stats[key]
                config = self.plot_configs.get(key, {"types": set(), "params": {}})
                print(f"plotting {key}")

                if isinstance(data, Dict):
                    # Check for min/avg/max pattern OR player-specific keys
                    has_min_max = all(k in data for k in ["min", "max"])
                    has_player_keys = any(
                        k.startswith("p") and k[1:].isdigit() for k in data.keys()
                    )

                    if (
                        has_min_max or has_player_keys
                    ) and PlotType.VARIATION_FILL in config["types"]:
                        self._plot_consolidated(ax, data, key, config)
                    else:
                        for subkey, subtensor in data.items():
                            print(f"  subkey {subkey}")
                            self._plot_tensor(ax, subtensor, f"{key}:{subkey}", config)
                else:
                    self._plot_tensor(ax, data, key, config)

                # Plot target line if exists as a horizontal line
                if key in self.targets and self.targets[key] is not None:
                    target_value = self.targets[key]
                    ax.axhline(
                        y=target_value,
                        color="r",
                        linestyle="--",
                        label=f"Target: {target_value}",
                    )
                    ax.legend()

            plt.tight_layout()
            if dir:
                save_dir = Path(dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{self.name}_stats.png"
                fig.savefig(save_path)
                print(f"Saved stats plot to {save_path.absolute()}")
            plt.close(fig)
        else:
            fig = None

        # Plot latent visualizations
        for key, data in self.latent_viz_data.items():
            print(f"plotting latent viz {key} using {data['method']}")
            method = data["method"].lower()
            latents = data["latents"]
            labels = data["labels"]
            kwargs = data["kwargs"]

            visualizer = None
            if method == "pca":
                visualizer = LatentPCAVisualizer(**kwargs)
            elif method == "tsne":
                visualizer = LatentTSNEVisualizer(**kwargs)
            elif method == "umap":
                if LatentUMAPVisualizer is None:
                    print(f"Skipping UMAP for {key}: umap-learn not installed.")
                    continue
                visualizer = LatentUMAPVisualizer(**kwargs)
            else:
                print(f"Unknown latent visualization method: {method}")
                continue

            if visualizer:
                save_path = None
                if dir:
                    save_dir = Path(dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{self.name}_{key}_{method}.png"

                # Check dimensionality before plotting
                # flatten if needed is handled by visualizer, but let's be safe on input type
                try:
                    visualizer.plot(
                        latents,
                        labels=labels,
                        save_path=save_path,
                        title=f"{self.name} - {key} ({method.upper()})",
                        show=False,
                    )
                except Exception as e:
                    print(f"Error plotting latent viz {key}: {e}")

        # Plot custom visualizations
        for key, data in self.custom_viz_data.items():
            print(f"plotting custom viz {key}")
            visualizer = data["visualizer"]
            viz_data = data["data"]
            kwargs = data["kwargs"]

            if visualizer and hasattr(visualizer, "plot"):
                save_path = None
                if dir:
                    save_dir = Path(dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{self.name}_{key}_custom.png"

                try:
                    visualizer.plot(
                        viz_data,
                        save_path=save_path,
                        title=f"{self.name} - {key}",
                        **kwargs,
                    )
                except Exception as e:
                    print(f"Error plotting custom viz {key}: {e}")

        if fig:
            plt.close(fig)
        return fig

    def _plot_consolidated(
        self, ax, data: Dict[str, List[Any]], label: str, config: Dict
    ):
        """Plots min/avg/max (or individual players) as a consolidated variation fill."""
        # Check for player-specific keys (p0, p1, p2, etc.)
        player_keys = sorted(
            [k for k in data.keys() if k.startswith("p") and k[1:].isdigit()]
        )
        player_data = {k: self._to_numpy(data[k]) for k in player_keys}

        avg_data = self._to_numpy(data.get("avg", data.get("score", [])))
        min_data = self._to_numpy(data.get("min", []))
        max_data = self._to_numpy(data.get("max", []))

        # If avg_data is empty, try to compute it from players
        if len(avg_data) == 0 and player_data:
            # Filter for players that actually have data
            active_players = {k: pd for k, pd in player_data.items() if len(pd) > 0}
            if active_players:
                # Find the minimum length among all players to allow stacking
                min_len = min(len(pd) for pd in active_players.values())
                stacked_players = np.stack(
                    [pd[:min_len] for pd in active_players.values()]
                )
                avg_data = np.mean(stacked_players, axis=0)

        if len(avg_data) == 0:
            return

        x = np.arange(len(avg_data))
        params = config["params"]
        if "x_scale" in params:
            x = x * params["x_scale"]

        # Plot player-specific lines first (fainter)
        for pk, pd in player_data.items():
            if len(pd) >= len(x):
                ax.plot(
                    x, pd[: len(x)], alpha=0.3, linestyle="--", label=f"{label} ({pk})"
                )

        # Plot main average line
        ax.plot(x, avg_data, label=f"{label} (avg)", linewidth=2)

        # Handle fill_between
        if len(min_data) >= len(x) and len(max_data) >= len(x):
            ax.fill_between(
                x,
                min_data[: len(x)],
                max_data[: len(x)],
                alpha=0.2,
                label=f"{label} (min-max)",
            )
        elif player_data:
            # If we have player data but no min/max keys, use min/max of players for fill
            active_players = [pd for pd in player_data.values() if len(pd) >= len(x)]
            if active_players:
                stacked_players = np.stack([pd[: len(x)] for pd in active_players])
                p_min = np.min(stacked_players, axis=0)
                p_max = np.max(stacked_players, axis=0)
                ax.fill_between(
                    x,
                    p_min,
                    p_max,
                    alpha=0.2,
                    label=f"{label} (P1-P{len(active_players)} fill)",
                )

        ax.set_title(f"{self.name} - {label}")
        ax.set_xlabel(params.get("x_label", "Updates"))
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _to_numpy(self, data: List[Any], reduce: bool = True) -> np.ndarray:
        if not data:
            return np.array([])
        if isinstance(data[0], torch.Tensor):
            if data[0].ndim > 0:
                tensor_data = torch.stack(data)
            else:
                tensor_data = torch.cat([d.view(-1) for d in data])
        else:
            # Filter and convert to tensors, handling potential non-convertible data
            try:
                # If we have a list of items, some might be dicts (improperly logged)
                # We attempt to convert what we can.
                valid_data = []
                for d in data:
                    if isinstance(d, (int, float, np.number, torch.Tensor)):
                        valid_data.append(d)
                    elif isinstance(d, np.ndarray):
                        valid_data.append(torch.from_numpy(d))
                    else:
                        # Skip or log error? Skipping for now to prevent crash
                        continue
                
                if not valid_data:
                    return np.array([])
                
                tensor_data = torch.as_tensor(valid_data)
            except (ValueError, TypeError, RuntimeError) as e:
                print(f"Warning: could not convert stat data to tensor: {e}")
                return np.array([])

        # Ensure we are on CPU and convert to numpy
        np_data = tensor_data.detach().cpu().numpy()

        # Squeeze singleton dimensions except the first one (batch/time dimension)
        if np_data.ndim > 1:
            # Keep first dim, squeeze others
            shape = list(np_data.shape)
            new_shape = [shape[0]] + [s for s in shape[1:] if s != 1]
            np_data = np_data.reshape(new_shape)

        # Reduce N-D tensors (N > 2) by averaging extra dimensions if reduce=True
        if reduce and np_data.ndim >= 2:
            # For 2D, this averages everything into a 1D line plot
            # For >2D, this collapses all feature dimensions into an average scalar per step
            axes_to_reduce = tuple(range(1, np_data.ndim))
            np_data = np.mean(np_data, axis=axes_to_reduce)

        return np_data

    def _plot_tensor(self, ax, data: List[Any], label: str, config: Dict):
        # Convert list to tensor for plotting if needed, or handle list directly
        if len(data) == 0:
            return

        types = config["types"]
        params = config["params"]

        # If BAR is present, we don't want to reduce the distribution to a single mean value
        np_data = self._to_numpy(data, reduce=(PlotType.BAR not in types))

        x = np.arange(len(np_data))

        # Handle x scaling
        if "x_scale" in params:
            x = x * params["x_scale"]
        if "x_start" in params or "x_end" in params:
            mask = np.ones_like(x, dtype=bool)
            if "x_start" in params:
                mask &= x >= params["x_start"]
            if "x_end" in params:
                mask &= x <= params["x_end"]
            x, np_data = x[mask], np_data[mask]

        if PlotType.BAR not in types:
            ax.plot(x, np_data, label=label)

        # Rolling average
        if PlotType.ROLLING_AVG in types:
            window = params.get("rolling_window", 10)
            if window > 1 and len(np_data) >= window:
                roll = np.convolve(np_data, np.ones(window) / window, mode="valid")
                ax.plot(x[window - 1 :], roll, label=f"{label} (rolling {window})")

        if PlotType.EXPONENTIAL_AVG in types:
            # The 'beta' parameter controls the smoothness of the EMA.
            # A common range is 0.9 to 0.999.
            # Higher beta means more smoothing (slower decay).
            beta = params.get("ema_beta", 0.9)

            if len(np_data) > 0:
                ema_data = np.zeros_like(np_data)
                v = 0.0  # Initial EMA value

                # TensorBoard often applies bias correction, but for simplicity
                # and common practice, we'll implement the standard EMA (v = beta*v + (1-beta)*datum)
                # and initialize v with the first data point.
                v = np_data[0]
                ema_data[0] = v

                for i in range(1, len(np_data)):
                    v = beta * v + (1 - beta) * np_data[i]
                    ema_data[i] = v

                ax.plot(
                    x, ema_data, label=f"{label} (EMA $\\beta={beta}$)", linestyle="-."
                )

        # Variation fill (std dev) - Only fallback if not using consolidated min/max
        if PlotType.VARIATION_FILL in types:
            mean = np.mean(np_data)
            std = np.std(np_data)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, label=f"{label} ±σ")

        # Best fit line
        if PlotType.BEST_FIT_LINE in types and len(x) > 1:
            coeffs = np.polyfit(x, np_data, 1)
            fit = np.polyval(coeffs, x)
            ax.plot(x, fit, linestyle="--", label=f"{label} fit")

        # Bar chart for the last step
        if PlotType.BAR in types:
            if len(np_data.shape) > 1:
                latest_data = np_data[-1]

                # Filter out actions with near-zero probability to reduce clutter
                threshold = params.get("bar_threshold", 0.01)
                significant_indices = np.where(latest_data > threshold)[0]

                # If too many are significant, just take the top N
                max_bars = params.get("max_bars", 20)
                if len(significant_indices) > max_bars:
                    top_indices = np.argsort(latest_data)[-max_bars:]
                    significant_indices = np.sort(top_indices)

                if len(significant_indices) == 0:
                    # Fallback to top 5 if nothing is "significant"
                    significant_indices = np.argsort(latest_data)[-5:]
                    significant_indices = np.sort(significant_indices)

                plot_data = latest_data[significant_indices]
                plot_x = np.arange(len(significant_indices))  # Dense X axis

                alpha = 0.5 if "network" in label or "search" in label else 0.8
                ax.bar(plot_x, plot_data, label=f"{label} (latest)", alpha=alpha)

                # Use the original significant_indices as labels for the dense x-axis
                ax.set_xticks(plot_x)
                ax.set_xticklabels([str(i) for i in significant_indices], rotation=45)
                ax.set_xlabel("Action Index")
            else:
                ax.bar(x, np_data, label=label, alpha=0.7)

        # Logarithmic scale
        if PlotType.LOG_Y in types:
            ax.set_yscale("log")

        ax.set_title(f"{self.name} - {label}")
        ax.set_xlabel(params.get("x_label", "Updates"))
        ax.set_ylabel(label)
        ax.grid()
        ax.legend()

    def get_data(self):
        if self._is_client:
            return None
        return self.stats
