import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- LOAD THE TRAINED MODEL AND SCALER ---
# Make sure these files ('logistic_model.joblib', 'robust_scaler.joblib')
# are in the same directory as this script.
try:
    model = joblib.load('logistic_model.joblib')
    scaler = joblib.load('robust_scaler.joblib')
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please make sure 'logistic_model.joblib' and 'robust_scaler.joblib' are in the correct folder.")
    st.stop()


# --- WEB APP INTERFACE ---
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write(
    "This app uses a Logistic Regression model to predict fraudulent credit card transactions. "
    "Input the transaction details below. The 'V' features are anonymized."
)
st.divider()

# --- USER INPUT SECTION ---
st.header("Transaction Details")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    time_val = st.number_input("Time (seconds since first transaction)", value=0.0)
with col2:
    amount_val = st.number_input("Amount (in currency)", value=100.0, format="%.2f")

# --- NEW TEXT BOX FOR V-FEATURES ---
# Replaced the 28 sliders with a single text area for comma-separated values.
st.subheader("Anonymized Features (V1-V28)")
v_features_string = st.text_area(
    "Enter all 28 'V' features as a comma-separated list (e.g., 0.1, -1.2, 0.5, ...)",
    # Default text to show the user the expected format
    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
)


# Button to trigger prediction
if st.button("Check for Fraud", type="primary"):
    
    # --- 1. PARSE AND VALIDATE V-FEATURES ---
    v_features = []
    try:
        # Split the string by comma, strip any extra whitespace, and convert to float
        v_features = [float(val.strip()) for val in v_features_string.split(',')]
        
        # Check if exactly 28 features were entered
        if len(v_features) != 28:
            st.error(f"Error: Expected 28 'V' features, but you entered {len(v_features)}. Please check your input.")
            st.stop() # Stop execution if the count is wrong

    except ValueError:
        st.error("Error: Could not parse 'V' features. Please make sure they are all numbers separated by commas (e.g., 1.2, -0.5, 3.4)")
        st.stop() # Stop execution if there's a non-numeric value
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

    # --- 2. PREPARE INPUT DATA ---
    try:
        # Scale 'Time' and 'Amount'
        # Note: This assumes the scaler was trained in a way that this transformation is correct.
        scaled_time = scaler.transform(np.array([[time_val]]))[0, 0]
        scaled_amount = scaler.transform(np.array([[amount_val]]))[0, 0]
        
        # Create the feature vector in the correct order
        # Order: scaled_amount, scaled_time, V1, V2, ..., V28
        features = [scaled_amount, scaled_time] + v_features
        input_data = pd.DataFrame([features])
        
        # --- 3. MAKE PREDICTION ---
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        st.divider()
        st.header("Prediction Result")
        
        if prediction[0] == 1:
            st.error("🚨 FRAUD DETECTED", icon="🚨")
            st.write(f"**Confidence Score:** {prediction_proba[0][1]*100:.2f}% probability of being fraudulent.")
        else:
            st.success("✅ Transaction appears to be GENUINE", icon="✅")
            st.write(f"**Confidence Score:** {prediction_proba[0][0]*100:.2f}% probability of being genuine.")

    except Exception as e:
        st.error(f"An error occurred during scaling or prediction: {e}")