"""Streamlit-based dashboard for RL observability.

Run with: streamlit run observability/dashboard/server.py

Plotly charts inside `plot_theme(...)` automatically pick up the active
theme (paper/plot background, font, colorway, axis colors). Streamlit's
native charts (st.line_chart etc.) use Vega-Lite and are NOT theme-bridged.
"""

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from observability.metrics.store import get_global_store
    from observability.plotting.style import plot_theme
except ImportError:
    st = None


def run_dashboard(theme: str = "matplotx:pitaya_smoothie"):
    if st is None:
        print("Streamlit/Plotly not found. Install with: pip install streamlit plotly")
        return

    st.set_page_config(page_title="RL-Stuff Observability", layout="wide")

    with plot_theme(theme):
        st.title("🚀 RL-Stuff Training Dashboard")

        store = get_global_store()

        st.sidebar.header("Configuration")
        refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 60, 5)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Actor Steps", "1.2M")
        with col2:
            st.metric("Learner Steps", "45K")
        with col3:
            st.metric("SPS", "3200")
        with col4:
            st.metric("UPS", "850")

        st.subheader("Training Curves")

        chart_data = pd.DataFrame(
            {"step": list(range(6)), "reward": [10, 20, 30, 40, 50, 60]}
        )
        fig = px.line(chart_data, x="step", y="reward", markers=True)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    run_dashboard()
