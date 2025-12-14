# ======================================
# TRAIN FLOOD RISK CLASSIFICATION MODEL
# ======================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================
# 1. Load Dataset
# ======================================
df = pd.read_csv("flood_risk_northern_mindanao_large.csv")
print("Dataset loaded successfully!")

# ======================================
# 2. Encode Categorical Columns
# ======================================
label_encoder = LabelEncoder()

# Encode Province
df["Province"] = label_encoder.fit_transform(df["Province"])

# Encode Target Label (Low / Medium / High)
df["Flood_Risk_Level"] = label_encoder.fit_transform(df["Flood_Risk_Level"])

# ======================================
# 3. Define Features and Target
# ======================================
X = df[[
    "Avg_Rainfall_mm",
    "River_Proximity_km",
    "Elevation_m",
    "Historical_Flood_Count",
    "Province"
]]

y = df["Flood_Risk_Level"]

# ======================================
# 4. Split the Data (80% Train, 20% Test)
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Data split completed (80% training, 20% testing).")

# ======================================
# 5. Train Random Forest
# ======================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

# ======================================
# 6. Predict & Evaluate
# ======================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n======================================")
print("        MODEL EVALUATION RESULTS       ")
print("======================================")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# ======================================
# 7. Save the Model
# ======================================
import pickle

model_filename = "flood_rf_model.pkl"

with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved successfully as {model_filename}")


