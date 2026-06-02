import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="Tri-Sensor Fusion Engine", layout="wide")

# --- UI HEADER ---
st.title("Advanced Tri-Sensor Fusion Engine")
st.markdown("Multi-modal target validation simulating Radar, LiDAR, and Optical sensors under environmental stress.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Operational Design Domain (ODD)")

weather = st.sidebar.radio(
    "Environmental Condition", 
    ["Clear", "Heavy Rain", "Thick Fog"],
    help="Clear: Nominal. Rain: Severely degrades LiDAR. Fog: Severely degrades Optical Camera."
)

clutter_index = st.sidebar.slider(
    "Ground Clutter (Radar Noise)", 0.0, 1.0, 0.3, 0.05,
    help="Simulates buildings/trees. High clutter increases the chance of Radar 'ghost' echoes."
)

altitude = st.sidebar.slider(
    "Target Altitude (m)", 10, 500, 150, 10,
    help="Higher altitudes reduce ground clutter interference but weaken LiDAR return strength."
)

# --- DATA GENERATION ENGINE ---
@st.cache_data
def generate_robust_data(weather_state, clutter, alt, points=500):
    np.random.seed(42)
    time_ms = np.arange(points) * 10
    x_pos = np.linspace(0, 1000, points) + np.random.normal(0, 2, points)
    y_pos = np.sin(x_pos / 100) * 50 + np.random.normal(0, 5, points)
    
    # Base probabilities
    base_radar = 0.8 - (clutter * 0.2)
    base_lidar = 1.0 - (alt / 800)
    base_camera = 0.9 - (alt / 1000)
    
    # Environmental Attenuation Modifiers
    mod_lidar = 1.0
    mod_camera = 1.0
    
    if weather_state == "Heavy Rain":
        mod_lidar = 0.3   # Rain destroys LiDAR pulses
        mod_camera = 0.7  # Rain blurs camera slightly
    elif weather_state == "Thick Fog":
        mod_lidar = 0.6   # Fog scatters LiDAR partially
        mod_camera = 0.1  # Fog completely blinds optical cameras
        
    # Generate noisy sensor streams
    radar_conf = np.clip(base_radar + np.random.uniform(-0.15, 0.25, points), 0.1, 1.0)
    lidar_conf = np.clip((base_lidar * mod_lidar) + np.random.uniform(-0.1, 0.1, points), 0.01, 1.0)
    camera_conf = np.clip((base_camera * mod_camera) + np.random.uniform(-0.1, 0.1, points), 0.01, 1.0)
    
    # Tri-Sensor Fusion Math (Weights: 40% Radar, 30% LiDAR, 30% Camera)
    fusion_score = (radar_conf * 0.40) + (lidar_conf * 0.30) + (camera_conf * 0.30)
    
    data = {
        "timestamp_ms": time_ms,
        "x_coordinate": x_pos,
        "y_coordinate": y_pos,
        "radar_conf": radar_conf,
        "lidar_conf": lidar_conf,
        "camera_conf": camera_conf,
        "fusion_score": fusion_score
    }
    return pd.DataFrame(data)

df = generate_robust_data(weather, clutter_index, altitude)

# --- METRICS & FILTERING ---
# Ghost Track: Radar is screaming (>0.7), but BOTH visual sensors are blind (<0.3)
ghost_tracks = df[(df["radar_conf"] > 0.7) & (df["lidar_conf"] < 0.3) & (df["camera_conf"] < 0.3)]
ghost_count = len(ghost_tracks)
avg_fusion = df["fusion_score"].mean() * 100

if ghost_count > 15:
    status, status_color = "CRITICAL: TARGETING COMPROMISED", "red"
elif ghost_count > 0:
    status, status_color = "WARNING: MARGINAL LOCK", "orange"
else:
    status, status_color = "NOMINAL: POSITIVE LOCK", "green"

# --- UI RENDERING ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Track Frames", f"{len(df):,}")
c2.metric("Ghost Tracks Detected", f"{ghost_count:,}", help="Instances where Radar hallucinates a target that LiDAR/Camera cannot see.")
c3.metric("System Fusion Health", f"{avg_fusion:.1f}%")
c4.markdown(f"### Status: :{status_color}[{status}]")

st.divider()

# --- VISUALIZATIONS ---
st.subheader("Live Multi-Modal Telemetry")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["x_coordinate"], df["y_coordinate"], color="#2c3e50", label="UAS Trajectory")
    if ghost_count > 0:
        ax.scatter(ghost_tracks["x_coordinate"], ghost_tracks["y_coordinate"], color="#e74c3c", label="Ghost Anomalies", zorder=5)
    ax.set_title("2D Spatial Tracking & Anomaly Coordinates")
    ax.set_xlabel("X Distance (m)")
    ax.set_ylabel("Y Deviation (m)")
    ax.legend()
    st.pyplot(fig)

with chart_col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["timestamp_ms"], df["radar_conf"], alpha=0.3, color="blue", label="Radar (RF)")
    ax.plot(df["timestamp_ms"], df["lidar_conf"], alpha=0.3, color="green", label="LiDAR (Laser)")
    ax.plot(df["timestamp_ms"], df["camera_conf"], alpha=0.3, color="orange", label="Camera (Optical)")
    ax.plot(df["timestamp_ms"], df["fusion_score"], color="black", linewidth=2.5, label="Fused Composite")
    ax.set_title("Tri-Sensor Confidence Degradation")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Confidence Matrix (0-1)")
    ax.legend(loc="lower right")
    st.pyplot(fig)

st.divider()

# --- DATA EXPORT ---
st.subheader("Edge Case Log (Actionable Debugging)")
if ghost_count > 0:
    st.dataframe(ghost_tracks.head(10), use_container_width=True)
else:
    st.success("Target lock stable across all modalities. Zero anomalies.")