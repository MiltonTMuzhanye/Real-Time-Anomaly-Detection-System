"""
Live Monitor Page - Real-time streaming visualization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import random

def render_live_monitor():
    """Render the live monitoring page"""
    st.title("🔴 Live Monitor")
    
    st.info("This page displays real-time data streams and anomaly detection in action")
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        speed = st.slider("Stream Speed", 1, 10, 5, help="Data points per second")
    
    with col2:
        show_anomalies = st.checkbox("Show Anomalies", value=True)
    
    with col3:
        if st.button("Start Streaming", type="primary"):
            st.session_state.streaming = True
    
    # Main streaming chart
    st.subheader("Live Data Stream")
    
    # Create placeholder for dynamic chart
    placeholder = st.empty()
    
    # Stats row
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    stats_placeholder1 = stats_col1.empty()
    stats_placeholder2 = stats_col2.empty()
    stats_placeholder3 = stats_col3.empty()
    stats_placeholder4 = stats_col4.empty()
    
    # Initialize data buffers
    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = []
        st.session_state.anomaly_buffer = []
        st.session_state.streaming = False
    
    # Streaming loop
    if st.session_state.streaming:
        try:
            for i in range(100):  # Limit to prevent infinite loop in demo
                # Generate new data point
                value = np.random.normal(50, 10) + 10 * np.sin(i / 20)
                
                # Occasionally generate anomaly
                is_anomaly = False
                if random.random() < 0.05:
                    value = value * 3
                    is_anomaly = True
                
                timestamp = datetime.now()
                
                # Add to buffer
                st.session_state.data_buffer.append({
                    'timestamp': timestamp,
                    'value': value,
                    'is_anomaly': is_anomaly
                })
                
                # Keep only last 100 points
                if len(st.session_state.data_buffer) > 100:
                    st.session_state.data_buffer.pop(0)
                
                # Update stats
                stats_placeholder1.metric(
                    "Current Value",
                    f"{value:.2f}",
                    delta=f"{value - 50:.2f}"
                )
                
                stats_placeholder2.metric(
                    "Anomalies Detected",
                    sum(1 for d in st.session_state.data_buffer if d['is_anomaly'])
                )
                
                stats_placeholder3.metric(
                    "Buffer Size",
                    len(st.session_state.data_buffer)
                )
                
                stats_placeholder4.metric(
                    "Stream Status",
                    "🟢 Active" if st.session_state.streaming else "🔴 Stopped"
                )
                
                # Create chart
                df = pd.DataFrame(st.session_state.data_buffer)
                
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("Data Stream", "Anomaly Detection"),
                    vertical_spacing=0.15
                )
                
                # Main data stream
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['value'],
                        mode='lines',
                        name='Data Stream',
                        line=dict(color='#00ff00', width=2)
                    ),
                    row=1,
                    col=1
                )
                
                # Anomaly points
                if show_anomalies:
                    anomaly_df = df[df['is_anomaly'] == True]
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_df['timestamp'],
                            y=anomaly_df['value'],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(color='red', size=15, symbol='x')
                        ),
                        row=1,
                        col=1
                    )
                
                # Anomaly scores (below)
                scores = [1.0 if d['is_anomaly'] else random.uniform(0, 0.5) for d in st.session_state.data_buffer]
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=scores,
                        mode='lines+markers',
                        name='Anomaly Score',
                        line=dict(color='#ff6600', width=2)
                    ),
                    row=2,
                    col=1
                )
                
                # Add threshold line
                fig.add_hline(
                    y=0.7,
                    line_dash="dash",
                    line_color="yellow",
                    annotation_text="Threshold",
                    row=2,
                    col=1
                )
                
                fig.update_layout(
                    height=600,
                    template='plotly_dark',
                    showlegend=True
                )
                
                placeholder.plotly_chart(fig, use_container_width=True)
                
                # Sleep based on speed
                time.sleep(1 / speed)
                
        except Exception as e:
            st.error(f"Streaming error: {e}")
            st.session_state.streaming = False
    
    else:
        st.info("Click 'Start Streaming' to begin live data monitoring")
    
    # Stop button
    if st.button("Stop Streaming"):
        st.session_state.streaming = False
        st.success("Streaming stopped")

if __name__ == "__main__":
    render_live_monitor()