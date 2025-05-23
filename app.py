import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("SVM_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Columns dropped during training
cols_to_drop = ['Inspection ID', 'DBA Name', 'AKA Name', 'License #', 'Address', 'City',
                'State', 'Zip', 'Location', 'Facility Type', 'Inspection Date', 'Violations']

# Function to preprocess new data to match training preprocessing
def preprocess_new_data(new_df, training_columns):
    # 1. Drop columns that were dropped in training
    new_df = new_df.drop(columns=[col for col in cols_to_drop if col in new_df.columns], errors='ignore')
    
    # 2. Drop rows with NA in target if target present (optional)
    if 'Results' in new_df.columns:
        new_df = new_df.dropna(subset=['Results'])
    new_df = new_df.dropna()
    
    # 3. Identify categorical columns except 'Results'
    categorical_cols = new_df.select_dtypes(include=['object']).columns.tolist()
    if 'Results' in categorical_cols:
        categorical_cols.remove('Results')
    
    # 4. Apply one-hot encoding, drop first to match training
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)
    
    # 5. Add missing columns that were in training but not in new_df_encoded, fill with zeros
    missing_cols = set(training_columns) - set(new_df_encoded.columns)
    for col in missing_cols:
        new_df_encoded[col] = 0
    
    # 6. Remove any extra columns not in training_columns (can happen if new categories appeared)
    extra_cols = set(new_df_encoded.columns) - set(training_columns)
    if extra_cols:
        new_df_encoded.drop(columns=list(extra_cols), inplace=True)
    
    # 7. Reorder columns to exactly match training columns
    new_df_encoded = new_df_encoded[training_columns]
    
    return new_df_encoded

# Load your original training dataset (to extract feature names)
# You need this because your model expects exactly those columns and order
train_df = pd.read_csv("/content/Food_Inspections_20250521.csv", engine="python", on_bad_lines="skip").sample(n=1000, random_state=42)
train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
train_df.dropna(subset=["Results"], inplace=True)
train_df.dropna(inplace=True)

categorical_cols_train = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols_train = [col for col in categorical_cols_train if col != 'Results']
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols_train, drop_first=True)

# Final training features (X columns)
training_columns = train_df_encoded.drop("Results", axis=1).columns.tolist()

# Load your new raw data that you want to predict on
new_data = pd.read_csv("/content/your_new_input.csv")  # Replace with your new file path

# Preprocess new data to match training features
X_new = preprocess_new_data(new_data, training_columns)

# Scale features
X_new_scaled = scaler.transform(X_new)

# Predict with the loaded model
predictions = model.predict(X_new_scaled)

# If you want prediction probabilities (if model supports it)
if hasattr(model, "predict_proba"):
    pred_probs = model.predict_proba(X_new_scaled)

# Map encoded labels back to original classes (optional)
# If you used LabelEncoder on 'Results' during training:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(train_df['Results'])  # Fit on training target to get classes
predicted_classes = label_encoder.inverse_transform(predictions)

print(predicted_classes)
