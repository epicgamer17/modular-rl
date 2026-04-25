import contextlib
import matplotlib as mpl
import matplotlib.pyplot as plt


_PLOTLY_TEMPLATE_NAME = "_observability_active_theme"


def _resolve_matplotx_style(spec: str, dark: bool):
    """Resolve a matplotx style spec to a flat rcParams dict.

    Accepts:
      "dracula"             - flat style, used as-is
      "pitaya_smoothie"     - nested group, picks variant via dark flag
      "tokyo_night.storm"   - nested group, explicit variant
    """
    import matplotx
    if "." in spec:
        name, variant = spec.split(".", 1)
    else:
        name, variant = spec, None

    style = getattr(matplotx.styles, name)
    if not isinstance(style, dict):
        raise AttributeError(f"matplotx.styles.{name} is not a style dict")

    is_nested = any(isinstance(v, dict) for v in style.values())
    if not is_nested:
        return style

    if variant is None:
        variant = "dark" if dark else "light"
    if variant not in style:
        raise KeyError(
            f"matplotx style '{name}' has no variant '{variant}'. "
            f"Available: {list(style.keys())}"
        )
    return style[variant]


def _apply_plotly_template_from_rc():
    """Mirror current matplotlib rcParams into a Plotly default template.

    Returns a teardown callable, or None if plotly isn't installed.
    Soft-fails on missing plotly so matplotlib-only callers aren't penalized.
    """
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except ImportError:
        return None

    rc = mpl.rcParams

    cycler = rc.get("axes.prop_cycle")
    colorway = list(cycler.by_key().get("color", [])) if cycler is not None else []

    font_family = rc.get("font.family")
    if isinstance(font_family, (list, tuple)) and font_family:
        font_family = font_family[0]
    if not isinstance(font_family, str):
        font_family = "sans-serif"

    axis_common = dict(
        gridcolor=rc.get("grid.color"),
        linecolor=rc.get("axes.edgecolor"),
        zerolinecolor=rc.get("axes.edgecolor"),
    )

    template = go.layout.Template(
        layout=dict(
            paper_bgcolor=rc.get("figure.facecolor"),
            plot_bgcolor=rc.get("axes.facecolor"),
            font=dict(color=rc.get("text.color"), family=font_family),
            colorway=colorway,
            xaxis=dict(
                **axis_common,
                tickcolor=rc.get("xtick.color"),
                tickfont=dict(color=rc.get("xtick.color")),
                title=dict(font=dict(color=rc.get("axes.labelcolor"))),
            ),
            yaxis=dict(
                **axis_common,
                tickcolor=rc.get("ytick.color"),
                tickfont=dict(color=rc.get("ytick.color")),
                title=dict(font=dict(color=rc.get("axes.labelcolor"))),
            ),
        )
    )

    prev_default = pio.templates.default
    pio.templates[_PLOTLY_TEMPLATE_NAME] = template
    pio.templates.default = _PLOTLY_TEMPLATE_NAME

    def restore():
        pio.templates.default = prev_default
        if _PLOTLY_TEMPLATE_NAME in pio.templates:
            del pio.templates[_PLOTLY_TEMPLATE_NAME]

    return restore


@contextlib.contextmanager
def plot_theme(theme: str = "matplotx:dracula", dark: bool = True):
    """Universal theme orchestrator with strict global state isolation.

    Applies the theme to matplotlib rcParams and (when plotly is installed)
    to a Plotly default template, so matplotlib figures and Plotly figures
    created inside the block share the same theme. Restores both on exit.

    Format examples:
    - "science+ieee"                  (SciencePlots, '+' joins style stack)
    - "catppuccin-mocha"              (Catppuccin)
    - "cyberpunk"                     (mplcyberpunk)
    - "matplotx:dracula"              (matplotx, flat style)
    - "matplotx:pitaya_smoothie"      (matplotx, nested - picks variant via `dark`)
    - "matplotx:tokyo_night.storm"    (matplotx, explicit nested variant)
    - "aquarel:arctic_light"          (aquarel)
    - "qbstyles"                      (qbstyles)
    - "ggplot"                        (matplotlib built-in)
    """
    original_rc = mpl.rcParams.copy()
    restore_plotly = None

    try:
        if theme.startswith("science"):
            import scienceplots  # noqa: F401
            plt.style.use(theme.split("+"))

        elif theme.startswith("matplotx:"):
            spec = theme.split(":", 1)[1]
            plt.style.use(_resolve_matplotx_style(spec, dark))

        elif theme.startswith("aquarel:"):
            from aquarel import Theme
            style_name = theme.split(":", 1)[1]
            Theme(style_name).apply()

        elif theme == "qbstyles":
            import qbstyles
            qbstyles.mpl_style(dark=dark)

        else:
            plt.style.use(theme)

        restore_plotly = _apply_plotly_template_from_rc()

        yield

    except ImportError as e:
        raise ImportError(f"Failed to load theme '{theme}'. Missing dependency: {e}")

    finally:
        if restore_plotly is not None:
            restore_plotly()
        mpl.rcParams.clear()
        mpl.rcParams.update(original_rc)


def get_alpha(uncertainty: bool = False) -> float:
    return 0.2 if uncertainty else 0.8
