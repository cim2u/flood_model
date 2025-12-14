import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# ===========================================
# PAGE SETTINGS
# ===========================================
st.set_page_config(
    page_title="Flood Risk Prediction – Northern Mindanao",
    layout="wide"
)

# ===========================================
# LOAD TRAINED MODEL
# ===========================================
with open("flood_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🌧 Flood Risk Prediction App – Northern Mindanao")
st.write("A Machine Learning Approach for Flood Risk Classification using Rainfall and River Data")

# ===========================================
# SESSION STATE – PREVENT REFRESH LOSS
# ===========================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None
    st.session_state.risk_text = None

# ===========================================
# MUNICIPALITIES (ACTUAL FROM YOUR DATASET)
# ===========================================
municipalities = {
    "Bukidnon": [
        "Maramag", "Baungon", "Talakag", "Manolo Fortich", "Malaybalay",
        "Kibawe", "Valencia City", "Libona", "Quezon"
    ],
    "Misamis Oriental": [
        "Balingasag", "Manticao", "Gingoog City", "Naawan", "Claveria",
        "Lugait", "Jasaan", "Opol", "El Salvador", "Villanueva",
        "Kinoguitan", "Cagayan de Oro", "Magsaysay", "Tagoloan", "Medina"
    ],
    "Misamis Occidental": [],  # No data provided yet
    "Lanao del Norte": ["Iligan City"],
    "Camiguin": []  # No data provided yet
}

# Approximate coordinates for municipalities
municipality_coords = {
    "Maramag": (7.76, 125.00),
    "Baungon": (8.32, 124.72),
    "Talakag": (8.23, 124.60),
    "Manolo Fortich": (8.36, 124.86),
    "Malaybalay": (8.15, 125.10),
    "Kibawe": (7.56, 124.98),
    "Valencia City": (7.90, 125.09),
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

# ===========================================
# USER INPUTS
# ===========================================
st.markdown("---")
st.subheader("Enter Environmental Data:")

col1, col2, col3 = st.columns(3)

with col1:
    avg_rainfall = st.number_input("Average Rainfall (mm)", min_value=0.0, format="%.2f")
    flood_count = st.number_input("Historical Flood Count", min_value=0, step=1)

with col2:
    river_distance = st.number_input("Distance to River (km)", min_value=0.0, format="%.2f")
    elevation = st.number_input("Elevation (meters above sea level)", min_value=0.0, format="%.2f")

with col3:
    st.subheader("Province:")
    province_list = list(municipalities.keys())
    province = st.selectbox("Select Province", province_list)

    # Update municipality dropdown dynamically
    location = st.selectbox(
        "Select Municipality / City",
        municipalities[province] if municipalities[province] else ["Not Available"]
    )

# Province encoding
province_mapping = {
    "Bukidnon": 0,
    "Misamis Oriental": 1,
    "Misamis Occidental": 2,
    "Lanao del Norte": 3,
    "Camiguin": 4
}
province_encoded = province_mapping[province]

# Risk labels
risk_labels = {0: "Low", 1: "Medium", 2: "High"}
risk_colors = {"Low": "green", "Medium": "orange", "High": "red"}

# ===========================================
# PREDICTION BUTTON
# ===========================================
st.markdown("---")
left, right = st.columns(2)

with left:
    if st.button("🔍 Predict Flood Risk", use_container_width=True):

        input_data = pd.DataFrame([[
            avg_rainfall,
            river_distance,
            elevation,
            flood_count,
            province_encoded
        ]], columns=[
            "Avg_Rainfall_mm",
            "River_Proximity_km",
            "Elevation_m",
            "Historical_Flood_Count",
            "Province"
        ])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        st.session_state.prediction = pred
        st.session_state.probability = prob
        st.session_state.risk_text = risk_labels[pred]

# Display result
if st.session_state.prediction is not None:
    st.subheader("Prediction Result:")
    st.write(f"### 🌊 Flood Risk Level: *{st.session_state.risk_text}*")

    st.subheader("Model Confidence:")
    st.write(f"Low Risk: {st.session_state.probability[0]:.2f}")
    st.write(f"Medium Risk: {st.session_state.probability[1]:.2f}")
    st.write(f"High Risk: {st.session_state.probability[2]:.2f}")

with right:
    st.info("""
    ### Flood Risk Levels:
    - *Low* – Minimal likelihood  
    - *Medium* – Possible flooding  
    - *High* – High likelihood of flood  
    """)

# ===========================================
# MAP SECTION
# ===========================================
st.markdown("---")
st.subheader("🗺️ Flood Zone Map – Northern Mindanao")

m = folium.Map(location=[8.5, 124.6], zoom_start=8)

if st.session_state.prediction is not None and location in municipality_coords:
    folium.Marker(
        location=municipality_coords[location],
        popup=f"{location}, {province} – Risk: {st.session_state.risk_text}",
        tooltip="Flood Prediction",
        icon=folium.Icon(color=risk_colors[st.session_state.risk_text])
    ).add_to(m)

st_folium(m, width=1000, height=450)

# ===========================================
# GRAPH
# ===========================================
st.markdown("---")
st.subheader("📈 Rainfall vs Predicted Flood Risk")

if st.session_state.prediction is not None:
    fig2, ax2 = plt.subplots()
    ax2.scatter(avg_rainfall, st.session_state.prediction, s=150)
    ax2.set_xlabel("Average Rainfall (mm)")
    ax2.set_ylabel("Flood Risk (0 = Low, 1 = Medium, 2 = High)")
    ax2.set_title("Rainfall and Flood Risk Relationship")
    st.pyplot(fig2)
else:
    st.info("Run a prediction to generate graph.")

# ===========================================
# FEATURE IMPORTANCE
# ===========================================
st.markdown("---")
st.subheader("📊 Feature Importance")

if st.checkbox("Show Feature Importance Chart"):
    importances = model.feature_importances_
    feature_names = [
        "Avg_Rainfall_mm",
        "River_Proximity_km",
        "Elevation_m",
        "Historical_Flood_Count",
        "Province"
    ]

    fig3, ax3 = plt.subplots()
    ax3.barh(feature_names, importances)
    ax3.set_xlabel("Importance Score")
    ax3.set_title("Feature Importance of Random Forest Model")
    st.pyplot(fig3)

    st.dataframe(pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False))







