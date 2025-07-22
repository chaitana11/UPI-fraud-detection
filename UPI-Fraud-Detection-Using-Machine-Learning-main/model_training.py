# model_training.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Step 1: Load Excel dataset
df = pd.read_excel("sample_fraud_dataset.xlsx")

# Step 2: Drop any missing values just in case
df.dropna(inplace=True)

# Step 3: Split features and label
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Step 4: One-hot encode categorical columns
X = pd.get_dummies(X)

# Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Use SMOTE with small sample safety
minority_count = sum(y == 1)
smote_k = 1 if minority_count < 6 else 5
smote = SMOTE(k_neighbors=smote_k, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 7: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Step 8: Train model with NumPy 2.x compatibility
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(
    np.asarray(X_train, dtype=np.float32),
    np.asarray(y_train)
)

# Step 9: Evaluate model
y_pred = model.predict(np.asarray(X_test, dtype=np.float32))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save model and scaler
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
