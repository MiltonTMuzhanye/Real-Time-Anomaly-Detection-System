import json
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

metrics_dir = Path("reports/metrics")
manifest_path = metrics_dir / "training_manifest.json"

if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    st.subheader("Trained Models")
    st.json(manifest)
else:
    st.info("No training manifest found. Run `python scripts/train.py` first.")

eval_path = metrics_dir / "evaluation.json"
if eval_path.exists():
    st.subheader("Latest Evaluation")
    st.json(json.loads(eval_path.read_text()))