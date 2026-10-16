import streamlit as st

st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Real-Time Anomaly Detection System")
st.caption("Numenta Anomaly Benchmark (NAB) — Streaming Detection & Monitoring")

st.markdown(
    """
### Pages

Use the sidebar to navigate:

- **Dashboard** — High-level KPIs and model comparison
- **Live Monitor** — Streaming detection in real time
- **Anomaly History** — Alert log and drill-down
- **Analytics** — NAB scoring, detection delay, threshold sweeps
"""
)