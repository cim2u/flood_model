# ======================================
# TRAIN FLOOD RISK CLASSIFICATION MODEL
# ======================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ======================================
# 1. Load Dataset
# ======================================
df = pd.read_csv("flood_risk_northern_mindanao_large.csv")
print("✅ Dataset loaded")

# ======================================
# 2. Encode Categorical Columns
# ======================================
le_province = LabelEncoder()
le_risk = LabelEncoder()

df["Province"] = le_province.fit_transform(df["Province"])
df["Flood_Risk_Level"] = le_risk.fit_transform(df["Flood_Risk_Level"])

# ======================================
# 3. Define Features & Target
# ======================================
X = df[
    [
        "Avg_Rainfall_mm",
        "River_Proximity_km",
        "Elevation_m",
        "Historical_Flood_Count",
        "Province"
    ]
]

y = df["Flood_Risk_Level"]

# ======================================
# 4. Train-Test Split
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Data split complete")

# ======================================
# 5. Train Random Forest Model
# ======================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained")

# ======================================
# 6. Evaluation
# ======================================
y_pred = model.predict(X_test)

print("\nMODEL ACCURACY:", accuracy_score(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))
print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))

# ======================================
# 7. Save Model
# ======================================
with open("flood_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as flood_rf_model.pkl")
