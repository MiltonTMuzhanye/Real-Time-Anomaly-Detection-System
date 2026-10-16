import json
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
st.title("📈 Analytics")

metrics_dir = Path("reports/metrics")
files = list(metrics_dir.glob("*.json"))

if not files:
    st.info("No metrics found. Run `scripts/evaluate.py` first.")
else:
    for f in files:
        with st.expander(f.name):
            st.json(json.loads(f.read_text()))

    sweep_path = metrics_dir / "threshold_sweep.json"
    if sweep_path.exists():
        df = pd.DataFrame(json.loads(sweep_path.read_text()))
        fig = px.line(df, x="threshold", y=["precision", "recall", "f1"],
                      title="Threshold sweep")
        st.plotly_chart(fig, use_container_width=True)