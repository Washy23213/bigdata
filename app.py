import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the trained model, scaler, and test data
model = joblib.load("SVM_best_model.pkl")
scaler = joblib.load("scaler.pkl")
X_test, y_test = joblib.load("test_data.pkl")  # Assumes (X_test, y_test) tuple

# Title
st.title("🍽 Food Inspection Results Predictor")

# Sidebar for user input mode
mode = st.sidebar.radio("Choose input mode", ["Manual Input", "CSV Upload"])

# Input form with example feature names
def get_user_input():
    st.subheader("Enter feature values")
    
    # Replace these with actual feature names used in your model
    inspection_score = st.number_input("Inspection Score", min_value=0.0, max_value=100.0, value=85.0)
    facility_type = st.selectbox("Facility Type", {"Grocery Store": 0, "Restaurant": 1})
    risk_level = st.selectbox("Risk Level", {"Low": 0, "Medium": 1, "High": 2})
    
    return pd.DataFrame([[inspection_score, facility_type, risk_level]],
                        columns=["Inspection_Score", "Facility_Type", "Risk_Level"])

# Prediction function
def predict(df):
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    return predictions

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

# Handle user input
if mode == "Manual Input":
    input_df = get_user_input()
    if st.button("Predict"):
        prediction = predict(input_df)
        st.success(f"Predicted Class: {prediction[0]}")

elif mode == "CSV Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview of uploaded data:")
        st.write(df.head())
        if st.button("Predict"):
            pred = predict(df)
            df["Prediction"] = pred
            st.success("Predictions completed!")
            st.write(df)

# Show model performance summary
if st.checkbox("Show Model Performance Summary"):
    show_confusion_matrix()
