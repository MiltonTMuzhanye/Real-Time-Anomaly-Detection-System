"""
Analytics Page - In-depth analysis and insights
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats

def render_analytics():
    """Render the analytics page"""
    st.title("📈 Analytics & Insights")
    
    # Time period selector
    period = st.selectbox(
        "Analysis Period",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"]
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 90
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Traffic patterns
    base_traffic = 1000 + 200 * np.sin(np.linspace(0, 4*np.pi, n_days))
    traffic_pattern = base_traffic + np.random.normal(0, 100, n_days)
    
    # Anomalies data
    anomaly_counts = np.random.poisson(5, n_days)
    anomaly_counts[anomaly_counts > 15] = 15
    anomaly_scores = np.random.uniform(0.3, 1.0, n_days)
    
    # Create metrics
    daily_data = pd.DataFrame({
        'Date': dates,
        'Traffic': traffic_pattern,
        'Anomalies': anomaly_counts,
        'AvgScore': anomaly_scores,
        'Critical': np.random.poisson(1, n_days),
        'High': np.random.poisson(2, n_days),
        'Medium': np.random.poisson(3, n_days),
        'Low': np.random.poisson(4, n_days)
    })
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Anomalies",
            daily_data['Anomalies'].sum(),
            delta=f"+{np.random.randint(1, 20)}%"
        )
    
    with col2:
        anomaly_rate = daily_data['Anomalies'].sum() / (n_days * 24)
        st.metric(
            "Avg Anomalies/Hour",
            f"{anomaly_rate:.2f}",
            delta=f"{np.random.uniform(-0.5, 0.5):.2f}"
        )
    
    with col3:
        peak_traffic = daily_data['Traffic'].max()
        st.metric(
            "Peak Traffic",
            f"{peak_traffic:,.0f}",
            delta=f"{peak_traffic - 1000:,.0f}"
        )
    
    with col4:
        avg_score = daily_data['AvgScore'].mean()
        st.metric(
            "Avg Anomaly Score",
            f"{avg_score:.3f}",
            delta=f"{avg_score - 0.5:.3f}"
        )
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Traffic vs Anomalies
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=daily_data['Date'],
                y=daily_data['Traffic'],
                name='Traffic Volume',
                line=dict(color='#00ff00', width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=daily_data['Date'],
                y=daily_data['Anomalies'],
                name='Anomalies',
                marker_color='red',
                opacity=0.6
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Traffic Volume vs Anomalies',
            template='plotly_dark',
            height=400
        )
        fig.update_yaxes(title_text="Traffic", secondary_y=False)
        fig.update_yaxes(title_text="Anomalies", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anomaly heatmap by day of week and hour
        heatmap_data = np.random.poisson(3, (7, 24))
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"{i}:00" for i in range(24)],
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='Reds'
        ))
        
        fig.update_layout(
            title='Anomaly Heatmap (Hour vs Day)',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical analysis
    st.subheader("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of anomaly scores
        scores = np.random.beta(2, 5, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=50,
            name='Anomaly Scores',
            marker_color='orange'
        ))
        
        fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                     annotation_text="Threshold")
        
        fig.update_layout(
            title='Anomaly Score Distribution',
            template='plotly_dark',
            height=400,
            xaxis_title='Score',
            yaxis_title='Frequency'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by severity
        severity_data = {
            'Critical': np.random.beta(2, 3, 50) + 0.1,
            'High': np.random.beta(3, 3, 100),
            'Medium': np.random.beta(3, 2, 150),
            'Low': np.random.beta(4, 2, 200)
        }
        
        fig = go.Figure()
        
        for severity, values in severity_data.items():
            fig.add_trace(go.Box(
                y=values,
                name=severity,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title='Score Distribution by Severity',
            template='plotly_dark',
            height=400,
            yaxis_title='Score'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.subheader("Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly trend with trend line
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Anomalies'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add trend line
        z = np.polyfit(range(len(daily_data)), daily_data['Anomalies'], 1)
        trend = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=daily_data['Date'],
            y=trend(range(len(daily_data))),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Anomaly Trend with Linear Regression',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation matrix
        corr_data = daily_data[['Traffic', 'Anomalies', 'AvgScore', 'Critical', 'High']]
        corr_matrix = corr_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Forecasting
    st.subheader("Anomaly Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Future predictions
        future_dates = pd.date_range(start=daily_data['Date'].iloc[-1], periods=30, freq='D')
        
        # Simple forecast
        avg_anomalies = daily_data['Anomalies'].mean()
        forecast = np.random.poisson(avg_anomalies, 30)
        confidence_lower = forecast * 0.8
        confidence_upper = forecast * 1.2
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_upper,
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(255,0,0,0)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_lower,
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.2)'
        ))
        
        fig.update_layout(
            title='Anomaly Forecast (Next 30 Days)',
            template='plotly_dark',
            height=400,
            xaxis_title='Date',
            yaxis_title='Predicted Anomalies'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Seasonality analysis
        seasonality = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Avg Anomalies': np.random.poisson(5, 7),
            'Avg Traffic': np.random.normal(1000, 200, 7)
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=seasonality['Day'],
                y=seasonality['Avg Anomalies'],
                name='Avg Anomalies',
                marker_color='red'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=seasonality['Day'],
                y=seasonality['Avg Traffic'],
                name='Avg Traffic',
                mode='lines+markers',
                line=dict(color='green', width=3)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Weekly Seasonality',
            template='plotly_dark',
            height=400,
            xaxis_title='Day of Week'
        )
        fig.update_yaxes(title_text="Anomalies", secondary_y=False)
        fig.update_yaxes(title_text="Traffic", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_analytics()