"""
Streamlit-based dashboard for RL observability.
Run with: streamlit run observability/dashboard/server.py
"""

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from observability.metrics.store import get_global_store
except ImportError:
    st = None

def run_dashboard():
    if st is None:
        print("Streamlit not found. Please install it with `pip install streamlit plotly`")
        return

    st.set_page_config(page_title="RL-Stuff Observability", layout="wide")
    st.title("🚀 RL-Stuff Training Dashboard")

    store = get_global_store()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 60, 5)

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Actor Steps", "1.2M") # Placeholder
    with col2:
        st.metric("Learner Steps", "45K") # Placeholder
    with col3:
        st.metric("SPS", "3200") # Placeholder
    with col4:
        st.metric("UPS", "850") # Placeholder

    # Charts
    st.subheader("Training Curves")
    
    # Example plot
    chart_data = pd.DataFrame(
       [10, 20, 30, 40, 50, 60],
       columns=['Reward']
    )
    st.line_chart(chart_data)

if __name__ == "__main__":
    run_dashboard()
