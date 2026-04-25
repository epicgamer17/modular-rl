import matplotlib.pyplot as plt
from matplotlib import ticker
from dataclasses import dataclass
from enum import Enum
import numpy as np
import seaborn as sns
from typing import Optional, List, Tuple
from .style import plot_theme, get_alpha


class SmoothingType(Enum):
    EMA = "ema"
    ROLLING = "rolling"
    NONE = "none"


class ConfidenceType(Enum):
    STD = "std"
    SE = "standard_error"
    IQR = "iqr"


@dataclass(frozen=True)
class DataConfig:
    smoothing: SmoothingType = SmoothingType.EMA
    window: int = 100
    include_confidence: bool = True
    confidence_type: ConfidenceType = ConfidenceType.IQR
    lower_percentile: int = 10
    upper_percentile: int = 90


def compute_ema(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    alpha = 2 / (window + 1)
    smoothed = np.empty(len(values))
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def compute_rolling_std(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) < window:
        return np.zeros(len(values))
    stds = np.array([np.std(values[max(0, i-window):i+1]) for i in range(len(values))])
    return stds


def compute_standard_error(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) < window:
        return np.zeros(len(values))
    stds = np.array([np.std(values[max(0, i-window):i+1]) for i in range(len(values))])
    n = np.array([min(window, i+1) for i in range(len(values))])
    return stds / np.sqrt(n)


def compute_rolling_percentiles(
    values: np.ndarray,
    window: int = 100,
    lower_pct: int = 10,
    upper_pct: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.zeros(len(values)), np.zeros(len(values))

    lower = np.array([
        np.percentile(values[max(0, i-window):i+1], lower_pct)
        for i in range(len(values))
    ])
    upper = np.array([
        np.percentile(values[max(0, i-window):i+1], upper_pct)
        for i in range(len(values))
    ])
    return lower, upper


def prepare_metric_data(
    raw_data: np.ndarray,
    data_config: Optional[DataConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if data_config is None:
        data_config = DataConfig()

    smoothing = data_config.smoothing
    window = data_config.window
    include_confidence = data_config.include_confidence
    confidence_type = data_config.confidence_type

    n = len(raw_data)
    x = np.arange(n)

    if smoothing == SmoothingType.EMA:
        smoothed = compute_ema(raw_data, window)
    elif smoothing == SmoothingType.ROLLING:
        kernel = np.ones(window) / window
        smoothed = np.convolve(raw_data, kernel, mode='same')
    else:
        smoothed = raw_data

    if include_confidence and n > window:
        if confidence_type == ConfidenceType.IQR:
            lower, upper = compute_rolling_percentiles(
                raw_data, window,
                data_config.lower_percentile,
                data_config.upper_percentile,
            )
        elif confidence_type == ConfidenceType.SE:
            spread = compute_standard_error(raw_data, window)
            lower = smoothed - spread
            upper = smoothed + spread
        else:
            spread = compute_rolling_std(raw_data, window)
            lower = smoothed - spread
            upper = smoothed + spread
    else:
        lower = None
        upper = None

    return x, smoothed, lower, upper


def _format_tick_labels(ax):
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    def format_x(value, pos):
        return f"{int(value):,}"

    def format_y(value, pos):
        if abs(value) >= 1000:
            return f"{value/1000:.1f}k"
        elif abs(value) < 1:
            return f"{value:.3f}"
        return f"{value:.1f}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_x))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y))


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _apply_cyberpunk_effects(ax):
    import mplcyberpunk
    mplcyberpunk.make_lines_glow(ax)
    mplcyberpunk.add_underglow(ax)


def plot_metric(
    name: Optional[str] = None,
    theme: str = "matplotx:pitaya_smoothie",
    title: Optional[str] = None,
    xlabel: str = "Training Step",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    show_confidence: bool = True,
    smoothed_data: Optional[np.ndarray] = None,
    raw_data: Optional[np.ndarray] = None,
    confidence_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """Plot a metric from the global metrics store or from explicit arrays.

    Usage by name:
        plot_metric("episode_reward", title="Return")

    Usage with explicit data:
        plot_metric(smoothed_data=..., raw_data=...)
    """
    if name is not None:
        from observability.metrics.store import get_global_store
        store = get_global_store()
        series = store.series(name)
        if series is None or len(series) == 0:
            raise ValueError(f"No data found for metric '{name}'")
        raw_data = np.array(series)
        _, smoothed_data, lower, upper = prepare_metric_data(raw_data)
        confidence_bounds = (lower, upper)

    if smoothed_data is None:
        raise ValueError("smoothed_data is required (pass `name` or `smoothed_data`)")

    resolved_title = title or (name.replace('_', ' ').title() if name else "")
    resolved_ylabel = ylabel or (name.replace('_', ' ').title() if name else "Value")

    with plot_theme(theme):
        x = np.arange(len(smoothed_data))

        fig, ax = plt.subplots(figsize=(10, 6))

        if raw_data is not None and len(raw_data) == len(x):
            ax.scatter(x, raw_data, color='C0', alpha=0.05, s=3, marker=".")

        line, = ax.plot(x, smoothed_data, color='C0', linewidth=2.5, label=resolved_ylabel)

        if confidence_bounds is not None and show_confidence:
            lower, upper = confidence_bounds
            if lower is not None and upper is not None:
                ax.fill_between(x, lower, upper, color='C0', alpha=get_alpha(True), linewidth=0)

        _format_tick_labels(ax)
        _despine(ax)

        ax.set_title(resolved_title or resolved_ylabel, fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel(resolved_ylabel, fontsize=14, fontweight="bold")
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=1,
            borderaxespad=0,
            frameon=False,
            fontsize=12,
        )
        ax.grid(True, alpha=0.15, linestyle=":")

        if theme == "cyberpunk":
            _apply_cyberpunk_effects(ax)

        annotations: List[object] = []

        def on_move(event):
            if event.inaxes != ax:
                return
            contains, inds = line.contains(event)
            if contains:
                ind = int(inds[0])
                x_val = float(x[ind])
                y_val = float(smoothed_data[ind])
                if annotations:
                    annotations[0].remove()
                    annotations.clear()
                annotations.append(ax.annotate(
                    f"Step: {int(x_val):,}\n{resolved_ylabel}: {y_val:.2f}",
                    xy=(x_val, y_val), fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
                ))
                fig.canvas.draw_idle()
            elif annotations:
                annotations[0].remove()
                annotations.clear()
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()


def plot_training_dashboard(
    metrics: Optional[List[str]] = None,
    data_dict: Optional[dict] = None,
    theme: str = "matplotx:pitaya_smoothie",
    save_path: Optional[str] = None,
):
    if data_dict is None or metrics is None:
        raise ValueError("metrics and data_dict are required")

    with plot_theme(theme):
        n = len(metrics)
        cols = 2
        rows = (n + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, name in enumerate(metrics):
            ax = axes[i]
            data = data_dict.get(name, (None, None, (None, None)))
            raw = data[0]
            smoothed = data[1]
            lower, upper = data[2] if len(data) > 2 else (None, None)

            x = np.arange(len(smoothed)) if smoothed is not None else np.array([])

            if raw is not None and len(raw) == len(x):
                ax.scatter(x, raw, color='C0', alpha=0.05, s=3, marker=".")

            if smoothed is not None:
                ax.plot(x, smoothed, color='C0', linewidth=2.5, label=name)

            if lower is not None and upper is not None:
                ax.fill_between(x, lower, upper, color='C0', alpha=get_alpha(True), linewidth=0)

            _format_tick_labels(ax)
            _despine(ax)
            ax.set_title(name.replace('_', ' ').title(), fontsize=14, fontweight="bold")
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel(name.replace('_', ' ').title(), fontsize=12)
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.01),
                ncol=1,
                borderaxespad=0,
                frameon=False,
                fontsize=10,
            )
            ax.grid(True, alpha=0.15, linestyle=":")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()


def plot_distribution(
    data: np.ndarray,
    name: Optional[str] = None,
    title: Optional[str] = None,
    theme: str = "matplotx:pitaya_smoothie",
):
    with plot_theme(theme):
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True, color='C1', alpha=0.6)
        label = name.replace('_', ' ').title() if name else "Value"
        plt.title(title or f"{label} Distribution")
        plt.show()
