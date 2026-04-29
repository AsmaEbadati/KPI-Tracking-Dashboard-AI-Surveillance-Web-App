
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Surveillance KPI Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# --- CUSTOM CSS FOR DARK MODE PROFESSIONALISM ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00ffcc; }
    div[data-testid="stMetricDelta"] { font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATION ENGINE ---
@st.cache_data
def get_historical_data():
    now = datetime.now()
    # Generating 100 data points for 4 cameras
    cameras = ['CAM-01 (Entrance)', 'CAM-02 (Parking)', 'CAM-03 (Loading)', 'CAM-04 (Lobby)']
    data_list = []
    
    for cam in cameras:
        for i in range(50):
            timestamp = now - timedelta(hours=i)
            # Simulate high accuracy with minor dips (model drift simulation)
            accuracy = np.random.uniform(0.92, 0.99) if i > 10 else np.random.uniform(0.84, 0.90)
            # Simulate latency (edge processing time)
            latency = np.random.uniform(150, 250) + (100 if i < 5 else 0)
            
            data_list.append({
                'Timestamp': timestamp,
                'Device_ID': cam,
                'Inference_Accuracy': accuracy,
                'Latency_ms': latency,
                'Uptime': 1 if np.random.random() > 0.02 else 0,
                'Object_Count': np.random.randint(10, 200)
            })
    return pd.DataFrame(data_list)

df = get_historical_data()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://img.icons8.com/fluency/96/security-camera.png", width=80)
st.sidebar.title("System Controls")
st.sidebar.markdown("---")
selected_cams = st.sidebar.multiselect("Active Cameras", df['Device_ID'].unique(), default=df['Device_ID'].unique())
date_filter = st.sidebar.date_input("Analysis Period", datetime.now() - timedelta(days=7))

st.sidebar.markdown("---")
st.sidebar.subheader("PM Portfolio Note")
st.sidebar.info("""
This dashboard tracks **Post-Deployment Health**. 
Key Focus: Identifying Model Drift and Edge Latency bottlenecks in B2B local environments.
""")

# --- MAIN CONTENT ---
st.title("🛡️ AI Video Analytics: Post-Deployment Monitor")
st.caption("Local B2B Surveillance Panel | Real-time Object Detection Pipeline")

filtered_df = df[df['Device_ID'].isin(selected_cams)]

# --- ROW 1: KEY PERFORMANCE INDICATORS ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    avg_acc = filtered_df['Inference_Accuracy'].mean()
    st.metric("Avg Inference Accuracy", f"{avg_acc:.1%}", delta="-1.2%" if avg_acc < 0.9 else "0.5%")
with m2:
    p95_latency = filtered_df['Latency_ms'].quantile(0.95)
    st.metric("P95 Latency (Edge)", f"{int(p95_latency)}ms", delta="15ms", delta_color="inverse")
with m3:
    uptime_pct = filtered_df['Uptime'].mean()
    st.metric("Fleet Uptime", f"{uptime_pct:.2%}")
with m4:
    total_detections = filtered_df['Object_Count'].sum()
    st.metric("Total Objects Tracked", f"{total_detections:,}")

st.markdown("---")

# --- ROW 2: VISUAL ANALYTICS ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Accuracy Trend (Model Reliability)")
    fig_acc = px.line(filtered_df, x='Timestamp', y='Inference_Accuracy', color='Device_ID',
                      template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_acc.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="SLA Bound")
    st.plotly_chart(fig_acc, use_container_width=True)

with col_right:
    st.subheader("Processing Latency Distribution")
    fig_lat = px.histogram(filtered_df, x='Latency_ms', nbins=30, 
                           template="plotly_dark", color_discrete_sequence=['#00ffcc'])
    st.plotly_chart(fig_lat, use_container_width=True)

# --- ROW 3: DEVICE UTILIZATION ---
st.subheader("Edge Device Resource Allocation")
util_cols = st.columns(len(selected_cams))
for i, cam in enumerate(selected_cams):
    with util_cols[i]:
        util = np.random.randint(65, 95)
        st.write(f"**{cam}**")
        st.progress(util)
        st.caption(f"GPU Load: {util}%")

# --- DATA TABLE VIEW ---
with st.expander("View Raw Performance Logs"):
    st.dataframe(filtered_df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
