import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ======================================
# PAGE SETTINGS
# ======================================
st.set_page_config(
    page_title="Flood Risk Classification – Northern Mindanao",
    layout="wide"
)

# ======================================
# CUSTOM CSS (MODERN UI)
# ======================================
st.markdown("""
<style>
body {
    background-color: #f4f6f8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.big-title {
    font-size: 48px;
    font-weight: 800;
    color: #1f4ed8;
    text-align: center;
    margin-bottom: 10px;
}
.subtitle {
    font-size: 20px;
    color: #555;
    text-align: center;
    margin-bottom: 30px;
}
.card {
    padding: 25px 30px;
    border-radius: 16px;
    background: linear-gradient(145deg, #ffffff, #e6f0ff);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    margin-bottom: 25px;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.12);
}
.result-low, .result-medium, .result-high {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    margin-top: 15px;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}
.result-low {
    color: #16a34a;
    background-color: #d1fae5;
}
.result-medium {
    color: #f59e0b;
    background-color: #fff7e0;
}
.result-high {
    color: #dc2626;
    background-color: #fee2e2;
}
.stButton>button {
    background-color: #1f4ed8;
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 12px 25px;
    border-radius: 12px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #3b82f6;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================
with open("flood_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# ======================================
# HEADER
# ======================================
st.markdown('<div class="big-title">🌧 Flood Risk Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Northern Mindanao • Survey-Based ML Model</div>', unsafe_allow_html=True)
st.markdown("---")

# ======================================
# SESSION STATE
# ======================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None
    st.session_state.risk_text = None

# ======================================
# LOCATION DATA
# ======================================
municipalities = {
    "Bukidnon": ["Maramag", "Baungon", "Talakag", "Manolo Fortich", "Malaybalay",
                 "Kibawe", "Valencia City", "Libona", "Quezon"],
    "Misamis Oriental": ["Balingasag", "Manticao", "Gingoog City", "Naawan", "Claveria",
                         "Lugait", "Jasaan", "Opol", "El Salvador", "Villanueva",
                         "Kinoguitan", "Cagayan de Oro", "Magsaysay", "Tagoloan", "Medina"],
    "Lanao del Norte": ["Iligan City"]
}

municipality_coords = {
    "Libona": (8.40, 124.73),
    "Quezon": (7.72, 125.10),
    "Balingasag": (8.75, 124.78),
    "Manticao": (8.40, 124.28),
    "Gingoog City": (8.83, 125.10),
    "Naawan": (8.43, 124.32),
    "Claveria": (8.62, 124.89),
    "Lugait": (8.33, 124.26),
    "Jasaan": (8.65, 124.75),
    "Opol": (8.52, 124.58),
    "El Salvador": (8.56, 124.52),
    "Villanueva": (8.58, 124.78),
    "Kinoguitan": (8.98, 124.79),
    "Cagayan de Oro": (8.48, 124.65),
    "Magsaysay": (8.96, 125.00),
    "Tagoloan": (8.53, 124.75),
    "Medina": (8.91, 125.02),
    "Iligan City": (8.23, 124.24)
}

province_mapping = {
    "Bukidnon": 0,
    "Misamis Oriental": 1,
    "Misamis Occidental": 2,
    "Lanao del Norte": 3,
    "Camiguin": 4
}

# ======================================
# SURVEY INPUTS
# ======================================
st.subheader("📝 Flood Risk Survey")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        rainfall_choice = st.radio(
            "🌧️ Has it been raining recently?",
            ["No rain", "Light rain", "Moderate rain", "Heavy rain", "Continuous rain"]
        )
        river_choice = st.radio(
            "🌊 River or canal condition",
            ["Normal", "Slightly rising", "High", "Near overflowing", "Overflowing"]
        )
    with col2:
        flood_prone_choice = st.radio(
            "🏞️ Is your area flood-prone?",
            ["Not flood-prone", "Sometimes floods", "Often floods", "Always floods"]
        )
        drainage_choice = st.radio(
            "🚰 Drainage condition",
            ["Good", "Slow", "Clogged", "Don't know"]
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================
# MAP INPUT MAPPING
# ======================================
rainfall_map = {"No rain":0,"Light rain":20,"Moderate rain":50,"Heavy rain":100,"Continuous rain":150}
river_map = {"Normal":5.0,"Slightly rising":3.0,"High":2.0,"Near overflowing":1.0,"Overflowing":0.5}
flood_map = {"Not flood-prone":0,"Sometimes floods":2,"Often floods":5,"Always floods":10}
elevation_map = {"Good":50,"Slow":30,"Clogged":10,"Don't know":25}

avg_rainfall = rainfall_map[rainfall_choice]
river_distance = river_map[river_choice]
flood_count = flood_map[flood_prone_choice]
elevation = elevation_map[drainage_choice]

# ======================================
# LOCATION SELECTION
# ======================================
st.subheader("📍 Location")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        province = st.selectbox("Province", list(municipalities.keys()))
    with col4:
        location = st.selectbox("Municipality / City", municipalities[province])
    st.markdown("</div>", unsafe_allow_html=True)
province_encoded = province_mapping[province]

# ======================================
# PREDICT BUTTON
# ======================================
st.markdown("---")
if st.button("🟧 Classify Flood Risk", use_container_width=True):
    input_df = pd.DataFrame(
        [[avg_rainfall, river_distance, elevation, flood_count, province_encoded]],
        columns=["Avg_Rainfall_mm","River_Proximity_km","Elevation_m","Historical_Flood_Count","Province"]
    )
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    st.session_state.prediction = pred
    st.session_state.probability = prob
    st.session_state.risk_text = risk_labels[pred]

# ======================================
# RESULTS DISPLAY
# ======================================
if st.session_state.prediction is not None:
    st.markdown("---")
    st.subheader("🎯 Classification Result")
    risk = st.session_state.risk_text
    if risk == "Low":
        st.markdown('<div class="result-low">LOW FLOOD RISK</div>', unsafe_allow_html=True)
    elif risk == "Medium":
        st.markdown('<div class="result-medium">MEDIUM FLOOD RISK</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-high">HIGH FLOOD RISK</div>', unsafe_allow_html=True)
    
    # Confidence Bars using Plotly
    st.markdown("### 📊 Model Confidence")
    fig = go.Figure(go.Bar(
        x=st.session_state.probability,
        y=["Low","Medium","High"],
        orientation='h',
        marker=dict(
            color=['#16a34a','#f59e0b','#dc2626']
        )
    ))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # MAP
    st.markdown("### 🗺️ Location Map")
    m = folium.Map(location=[8.5, 124.6], zoom_start=8)
    if location in municipality_coords:
        folium.Marker(
            location=municipality_coords[location],
            popup=f"{location} – {risk}",
            icon=folium.Icon(color="red" if risk=="High" else "orange" if risk=="Medium" else "green")
        ).add_to(m)
    st_folium(m, width=1000, height=450)
