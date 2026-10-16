import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Anomaly History", page_icon="📜", layout="wide")
st.title("📜 Anomaly History")

log_path = Path("reports/anomaly_reports/alerts.jsonl")
if log_path.exists():
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No alerts logged yet.")
else:
    st.info("No alert log found. Run `scripts/stream.py` and pipe alerts to reports/anomaly_reports/alerts.jsonl.")