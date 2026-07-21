"""
Anomaly History Page - Historical analysis and filtering
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

def render_anomaly_history():
    """Render the anomaly history page"""
    st.title("📜 Anomaly History")
    
    # Filters
    st.subheader("Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            [datetime.now() - timedelta(days=30), datetime.now()]
        )
    
    with col2:
        severity_filter = st.multiselect(
            "Severity",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"]
        )
    
    with col3:
        type_filter = st.multiselect(
            "Anomaly Type",
            ["DDoS", "Scanning", "Misconfiguration", "Flash Crowd", "Unknown"]
        )
    
    with col4:
        status_filter = st.multiselect(
            "Status",
            ["Active", "Acknowledged", "Resolved", "False Positive"]
        )
    
    # Generate sample history data
    np.random.seed(42)
    n_samples = 100
    
    history_data = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=n_samples, freq='6h'),
        'Severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], n_samples),
        'Score': np.random.uniform(0.3, 1.0, n_samples),
        'Type': np.random.choice(['DDoS', 'Scanning', 'Misconfiguration', 'Flash Crowd'], n_samples),
        'Status': np.random.choice(['Active', 'Acknowledged', 'Resolved'], n_samples),
        'Detected By': np.random.choice(['Ensemble', 'Isolation Forest', 'Autoencoder'], n_samples),
        'Response Time': np.random.exponential(100, n_samples),
        'IP Address': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'Description': [f"Anomaly detected in network traffic" for _ in range(n_samples)]
    })
    
    # Apply filters
    filtered_data = history_data.copy()
    
    if severity_filter:
        filtered_data = filtered_data[filtered_data['Severity'].isin(severity_filter)]
    
    if type_filter:
        filtered_data = filtered_data[filtered_data['Type'].isin(type_filter)]
    
    if status_filter:
        filtered_data = filtered_data[filtered_data['Status'].isin(status_filter)]
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anomalies", len(filtered_data))
    
    with col2:
        critical = len(filtered_data[filtered_data['Severity'] == 'Critical'])
        st.metric("Critical", critical)
    
    with col3:
        resolved = len(filtered_data[filtered_data['Status'] == 'Resolved'])
        st.metric("Resolved", resolved)
    
    with col4:
        avg_score = filtered_data['Score'].mean()
        st.metric("Avg Score", f"{avg_score:.3f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly trend over time
        daily_data = filtered_data.groupby(filtered_data['Timestamp'].dt.date).size().reset_index(name='Count')
        
        fig = px.line(
            daily_data,
            x='Timestamp',
            y='Count',
            title='Anomaly Trend Over Time',
            line_shape='spline'
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_counts = filtered_data['Severity'].value_counts()
        
        fig = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title='Severity Distribution',
            hole=0.3
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Anomaly Details")
    
    # Search box
    search = st.text_input("Search anomalies...")
    
    if search:
        mask = filtered_data['Description'].str.contains(search, case=False) | \
               filtered_data['IP Address'].str.contains(search, case=False)
        display_data = filtered_data[mask]
    else:
        display_data = filtered_data
    
    # Color coding for severity
    def color_severity(val):
        colors = {
            'Critical': '#ff0000',
            'High': '#ff6600',
            'Medium': '#ffcc00',
            'Low': '#00cc00'
        }
        return f'background-color: {colors.get(val, "white")}'
    
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export to CSV"):
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"anomaly_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export to JSON"):
            json_data = display_data.to_json(orient='records')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"anomaly_history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    # Detailed view
    st.subheader("Anomaly Details")
    selected_row = st.selectbox(
        "Select an anomaly to view details",
        display_data.index if not display_data.empty else []
    )
    
    if not display_data.empty and selected_row is not None:
        details = display_data.iloc[selected_row]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                'Timestamp': str(details['Timestamp']),
                'Severity': details['Severity'],
                'Type': details['Type'],
                'Score': details['Score']
            })
        
        with col2:
            st.json({
                'Status': details['Status'],
                'Detected By': details['Detected By'],
                'Response Time': f"{details['Response Time']:.0f}ms",
                'IP Address': details['IP Address']
            })

if __name__ == "__main__":
    render_anomaly_history()