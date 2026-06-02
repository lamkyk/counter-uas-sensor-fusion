import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="Sensor Fusion Evaluator", layout="wide")

# --- UI HEADER ---
st.title("Counter-UAS Sensor Fusion Evaluator")
st.markdown("Validate target lock stability and filter ghost tracks across multi-modal sensor inputs.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Operational Design Domain")
clutter_index = st.sidebar.slider("Environmental Clutter Index", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
altitude = st.sidebar.slider("Target Altitude (m)", min_value=10, max_value=500, value=150, step=10)
radar_cross_section = st.sidebar.slider("Radar Cross Section (m^2)", min_value=0.01, max_value=5.0, value=0.1, step=0.01)

# --- DATA GENERATION ---
@st.cache_data
def generate_fusion_data(clutter, alt, rcs, points=500):
    np.random.seed(42)
    
    # Base kinematics
    time_ms = np.arange(points) * 10
    x_pos = np.linspace(0, 1000, points) + np.random.normal(0, 2, points)
    y_pos = np.sin(x_pos / 100) * 50 + np.random.normal(0, 5, points)
    
    # Sensor simulations
    radar_conf = np.clip((rcs * 0.8) - (clutter * 0.2) + np.random.uniform(-0.1, 0.2, points), 0.1, 1.0)
    lidar_conf = np.clip((1.0 - (alt / 600)) - (clutter * 0.4) + np.random.uniform(-0.2, 0.1, points), 0.05, 1.0)
    
    # Fusion logic
    fusion_score = (radar_conf * 0.6) + (lidar_conf * 0.4)
    
    data = {
        "timestamp_ms": time_ms,
        "x_coordinate": x_pos,
        "y_coordinate": y_pos,
        "radar_confidence": radar_conf,
        "lidar_confidence": lidar_conf,
        "fusion_score": fusion_score
    }
    return pd.DataFrame(data)

df = generate_fusion_data(clutter_index, altitude, radar_cross_section)

# --- METRICS PROCESSING ---
ghost_tracks = df[(df["radar_confidence"] > 0.7) & (df["lidar_confidence"] < 0.2)]
ghost_count = len(ghost_tracks)
avg_fusion = df["fusion_score"].mean() * 100

if ghost_count > 20:
    status = "TARGETING COMPROMISED"
    status_color = "red"
elif ghost_count > 0:
    status = "MARGINAL LOCK"
    status_color = "orange"
else:
    status = "POSITIVE LOCK"
    status_color = "green"

# --- EXECUTIVE SUMMARY CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Track Points", f"{len(df):,}")
col2.metric("Ghost Tracks (Edge Cases)", f"{ghost_count:,}")
col3.metric("Avg Fusion Confidence", f"{avg_fusion:.1f}%")
col4.markdown(f"### Lock Status: :{status_color}[{status}]")

st.divider()

# --- VISUALIZATIONS ---
st.subheader("Live Kinematics & Sensor Fusion")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["x_coordinate"], df["y_coordinate"], color="#27ae60", label="UAS Trajectory")
    if ghost_count > 0:
        ax.scatter(ghost_tracks["x_coordinate"], ghost_tracks["y_coordinate"], color="#c0392b", label="Ghost Track Anomalies", zorder=5)
    ax.set_title("2D Spatial Tracking")
    ax.set_xlabel("X Distance (m)")
    ax.set_ylabel("Y Deviation (m)")
    ax.legend()
    st.pyplot(fig)

with chart_col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["timestamp_ms"], df["radar_confidence"], alpha=0.7, label="Radar")
    ax.plot(df["timestamp_ms"], df["lidar_confidence"], alpha=0.7, label="LiDAR")
    ax.plot(df["timestamp_ms"], df["fusion_score"], color="black", linewidth=2, label="Fused Output")
    ax.set_title("Multimodal Confidence Over Time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Confidence Score")
    ax.legend()
    st.pyplot(fig)

st.divider()

# --- DATA TABLE ---
st.subheader("High-Uncertainty ODD Edge Cases")
if ghost_count > 0:
    st.dataframe(ghost_tracks.head(10), use_container_width=True)
else:
    st.success("Target lock stable. Zero edge case anomalies detected.")