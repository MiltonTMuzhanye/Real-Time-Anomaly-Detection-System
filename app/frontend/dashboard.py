"""
Dashboard Page - Main monitoring dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import numpy as np

def render_dashboard():
    """Render the main dashboard"""
    st.title("📊 Anomaly Detection Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Anomalies",
            "1,247",
            delta="+12%",
            help="Total anomalies detected"
        )
    
    with col2:
        st.metric(
            "Detection Rate",
            "98.5%",
            delta="+0.3%",
            help="Percentage of anomalies correctly detected"
        )
    
    with col3:
        st.metric(
            "False Positives",
            "89",
            delta="-5",
            help="Number of false positive alerts"
        )
    
    with col4:
        st.metric(
            "Avg Response Time",
            "32ms",
            delta="-3ms",
            help="Average detection response time"
        )
    
    with col5:
        st.metric(
            "Model Accuracy",
            "96.8%",
            delta="+0.5%",
            help="Overall model accuracy"
        )
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Real-time Anomaly Detection")
        
        # Create real-time chart
        fig = go.Figure()
        
        # Generate sample data
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq='1s')
        values = np.random.normal(0, 1, 100).cumsum() + 50
        anomalies = np.random.choice([0, 1], size=100, p=[0.95, 0.05])
        anomaly_points = values * anomalies
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name='Traffic Volume',
            line=dict(color='#00ff00', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps[anomalies == 1],
            y=values[anomalies == 1],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=12, symbol='x')
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Anomaly Distribution")
        
        # Create pie chart
        labels = ['Critical', 'High', 'Medium', 'Low']
        values = [12, 25, 30, 18]
        colors = ['#ff0000', '#ff6600', '#ffcc00', '#00cc00']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=colors)
        )])
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            title='Anomalies by Severity'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Performance")
        
        # Performance metrics
        metrics_data = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Value': [0.94, 0.89, 0.92, 0.97]
        })
        
        fig = px.bar(
            metrics_data,
            x='Metric',
            y='Value',
            title='Model Performance Metrics',
            color='Metric',
            range_y=[0.8, 1.0]
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recent Alerts")
        
        # Sample alerts table
        alerts = pd.DataFrame({
            'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='5min'),
            'Severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], 10),
            'Score': np.random.uniform(0.5, 1.0, 10),
            'Source': np.random.choice(['Network', 'System', 'Application'], 10),
            'Status': np.random.choice(['Active', 'Acknowledged', 'Resolved'], 10)
        })
        
        st.dataframe(alerts, use_container_width=True, height=200)

if __name__ == "__main__":
    render_dashboard()