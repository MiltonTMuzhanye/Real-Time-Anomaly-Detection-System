import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly_detection.data.ingestion import load_nab_series
from anomaly_detection.features.rolling_features import SlidingWindowFeatureExtractor
from app.inference.anomaly_engine import AnomalyEngine
from app.inference.alert_engine import AlertEngine

st.set_page_config(page_title="Live Monitor", page_icon="📡", layout="wide")
st.title("📡 Live Monitor")

MODEL_PATH = "artifacts/trained_models/isolation_forest.joblib"
DEFAULT_SERIES = "data/raw/nab/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"

col1, col2, col3 = st.columns(3)
threshold = col1.slider("Threshold", 0.0, 1.0, 0.65, 0.01)
speed = col2.slider("Replay speed (x)", 1, 200, 50)
series_path = col3.text_input("Series path", DEFAULT_SERIES)

start = st.button("▶ Start Streaming")
stop = st.button("⏹ Stop")

placeholder = st.empty()
stats_box = st.empty()

if start:
    detector = joblib.load(MODEL_PATH)
    extractor = SlidingWindowFeatureExtractor(window_size=128, stride=1)
    alerts = AlertEngine(cooldown_events=10)
    engine = AnomalyEngine(extractor, detector, threshold, alerts)

    df = load_nab_series(series_path)
    values, times, scores = [], [], []

    for i, row in df.iterrows():
        result = engine.ingest({"timestamp": row["timestamp"].isoformat(), "value": float(row["value"])})
        values.append(float(row["value"]))
        times.append(row["timestamp"])
        scores.append(result["score"] if result else 0.0)

        if i % max(1, int(speed // 5)) == 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name="value"))
            anom_x = [t for t, s in zip(times, scores) if s > threshold]
            anom_y = [v for v, s in zip(values, scores) if s > threshold]
            fig.add_trace(go.Scatter(x=anom_x, y=anom_y, mode="markers",
                                     marker=dict(color="red", size=8), name="anomaly"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            placeholder.plotly_chart(fig, use_container_width=True)
            stats_box.metric("Alerts", engine.stats["total_alerts"])
            time.sleep(0.05)
        if stop:
            break

    st.success(f"Stream ended. {engine.stats}")