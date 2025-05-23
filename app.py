import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Load model, scaler, test data
model = joblib.load("SVM_best_model.pkl")
scaler = joblib.load("scaler.pkl")
X_test, y_test = joblib.load("test_data.pkl")

# Title
st.title("🍽 Food Inspection Results Predictor")

# Sidebar for input mode
mode = st.sidebar.radio("Choose input mode", ["Manual Input", "CSV Upload"])

# Helper: map categorical strings to integers (use your training mappings)
inspection_type_map = {"Type A": 0, "Type B": 1, "Type C": 2}
facility_type_map = {"Grocery Store": 0, "Restaurant": 1, "Food Stand": 2}
risk_map = {"Low": 0, "Medium": 1, "High": 2}

def preprocess_input(df):
    # Map categorical to int
    df["Inspection Type"] = df["Inspection Type"].map(inspection_type_map)
    df["Facility Type"] = df["Facility Type"].map(facility_type_map)
    df["Risk"] = df["Risk"].map(risk_map)
    
    # Convert Inspection Date to datetime
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"])
    
    # Extract numeric date features
    df["Inspection_Year"] = df["Inspection Date"].dt.year
    df["Inspection_Month"] = df["Inspection Date"].dt.month
    df["Inspection_Day"] = df["Inspection Date"].dt.day
    
    # Drop original date column after extraction
    df = df.drop(columns=["Inspection Date"])
    
    # Reorder columns to match scaler and model input (IMPORTANT!)
    features_order = ["Inspection Type", "Risk", "Facility Type",
                      "Inspection_Year", "Inspection_Month", "Inspection_Day"]
    
    df = df[features_order]
    return df

# Manual input form
def get_user_input():
    st.subheader("Enter feature values")
    
    inspection_type = st.selectbox("Inspection Type", list(inspection_type_map.keys()))
    inspection_date = st.date_input("Inspection Date")
    risk = st.selectbox("Risk Level", list(risk_map.keys()))
    facility_type = st.selectbox("Facility Type", list(facility_type_map.keys()))
    
    data = {
        "Inspection Type": [inspection_type],
        "Inspection Date": [inspection_date],
        "Risk": [risk],
        "Facility Type": [facility_type],
    }
    df = pd.DataFrame(data)
    df_processed = preprocess_input(df)
    return df_processed

# Prediction function
def predict(df):
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    return preds

# Show confusion matrix and accuracy
def show_confusion_matrix():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.subheader(f"Model Accuracy: {acc:.4f}")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Main app logic
if mode == "Manual Input":
    input_df = get_user_input()
    if st.button("Predict"):
        prediction = predict(input_df)
        st.success(f"Predicted Class: {prediction[0]}")

elif mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.write(df.head())
        if st.button("Predict"):
            df_processed = preprocess_input(df)
            preds = predict(df_processed)
            df["Prediction"] = preds
            st.success("Predictions completed!")
            st.write(df)

if st.checkbox("Show Model Performance Summary"):
    show_confusion_matrix()
