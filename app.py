import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the trained model and scaler
model = joblib.load("RandomForest_best_model.pkl")  # Replace with your best model filename
scaler = joblib.load("scaler.pkl")  # Save and load this from your training script

# Title
st.title("🍽 Food Inspection Results Predictor")

# Sidebar for user input mode
mode = st.sidebar.radio("Choose input mode", ["Manual Input", "CSV Upload"])

# Input form
def get_user_input():
    st.subheader("Enter feature values")
    # Customize this with your actual feature names and types
    feature1 = st.number_input("Feature 1")
    feature2 = st.number_input("Feature 2")
    feature3 = st.selectbox("Feature 3 (categorical)", [0, 1])  # Example dummy value
    return pd.DataFrame([[feature1, feature2, feature3]], columns=["Feature1", "Feature2", "Feature3"])

# Prediction
def predict(df):
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    return predictions

# Accuracy
def load_test_data():
    return joblib.load("test_data.pkl")  # Save (X_test, y_test) in training script for evaluation here

def show_confusion_matrix():
    X_test, y_test = load_test_data()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader(f"Model Accuracy: {acc:.4f}")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Input Handling
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

# Show performance metrics
if st.checkbox("Show Model Performance Summary"):
    show_confusion_matrix()
