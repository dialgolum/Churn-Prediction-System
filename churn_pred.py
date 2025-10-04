import os
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Load the trained model and scaler with graceful errors
model = None
scaler = None
model_path = "best_model.pkl"
scaler_path = "scaler.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
else:
    st.error(f"Model file not found: {model_path}")

if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load scaler '{scaler_path}': {e}")
else:
    st.error(f"Scaler file not found: {scaler_path}")

# Define the input features for the model
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0.0", "HasCrCard_1.0",
    "IsActiveMember_0.0", "IsActiveMember_1.0"
]

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Updated default values
default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False, False, True, False, True
]

# Sidebar setup
# Sidebar image (use exact filename from repo; filenames are case-sensitive on deploy)
if os.path.exists("Pic 1.PNG"):
    st.sidebar.image("Pic 1.PNG", use_column_width=True)
else:
    # fallback: don't crash if image missing
    st.sidebar.write("")
st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}
for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1 if isinstance(default_values[i], int) else 0.01
        )
    elif isinstance(default_values[i], bool):
        user_inputs[feature] = st.sidebar.checkbox(feature, value=default_values[i])
    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1
        )

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler to the required columns
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# App Header
if os.path.exists("Pic 2.PNG"):
    st.image("Pic 2.PNG", use_column_width=True)
else:
    st.title("Customer Churn Prediction")
st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
with left_col:
    st.header("Feature Importance")
    # Load feature importance data from the Excel file
    feature_importance_df = None
    fi_path = "feature_importance.xlsx"
    if os.path.exists(fi_path):
        try:
            feature_importance_df = pd.read_excel(fi_path, usecols=["Feature", "Feature Importance Score"])
        except Exception as e:
            st.error(f"Failed to read feature importance file '{fi_path}': {e}")
    else:
        st.error(f"Feature importance file not found: {fi_path}")

    if feature_importance_df is not None:
        # Plot the feature importance bar chart
        fig = px.bar(
            feature_importance_df.sort_values(by="Feature Importance Score", ascending=True),
            x="Feature Importance Score",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            labels={"Feature Importance Score": "Importance", "Feature": "Features"},
            width=400,  # Set custom width
            height=500  # Set custom height
        )
        st.plotly_chart(fig)

# Right Page: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        if model is None or scaler is None:
            st.error("Model or scaler not loaded. Cannot run prediction.")
        else:
            # Get the predicted probabilities and label
            try:
                probabilities = model.predict_proba(input_data_scaled)[0]
                prediction = model.predict(input_data_scaled)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                probabilities = None
                prediction = None
            if prediction is not None:
                # Map prediction to label
                prediction_label = "Churned" if prediction == 1 else "Retain"

                # Display results
                st.subheader(f"Predicted Value: {prediction_label}")
                st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
                st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
                # Display a clear output for the prediction
                st.markdown(f"### Output: **{prediction_label}**")

# Streamlit run churn_pred.py