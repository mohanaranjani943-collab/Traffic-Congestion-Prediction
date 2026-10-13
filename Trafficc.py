import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE TITLE
# ============================================================

st.title("🚦 Traffic Congestion Prediction")

st.write("Enter the details below to predict the number of vehicles.")

# ============================================================
# LOAD SAVED MODEL
# ============================================================

model = joblib.load(
    r"C:\Users\ELCOT\Downloads\Mona\traffic_model.pkl"
)

# ============================================================
# USER INPUTS
# ============================================================

year = st.number_input(
    "Enter Year",
    min_value=2020,
    max_value=2030,
    value=2026,
    step=1
)

month = st.number_input(
    "Enter Month",
    min_value=1,
    max_value=12,
    value=10,
    step=1
)

day = st.number_input(
    "Enter Day",
    min_value=1,
    max_value=31,
    value=7,
    step=1
)

hour = st.number_input(
    "Enter Hour",
    min_value=0,
    max_value=23,
    value=10,
    step=1
)

day_of_week = st.number_input(
    "Enter Day of Week",
    min_value=0,
    max_value=6,
    value=2,
    step=1
)

weekend = st.number_input(
    "Weekend (0 = No, 1 = Yes)",
    min_value=0,
    max_value=1,
    value=0,
    step=1
)

junction = st.number_input(
    "Enter Junction",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)

# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Traffic"):

    # Create input data
    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Hour": [hour],
        "DayOfWeek": [day_of_week],
        "Weekend": [weekend],
        "Junction": [junction]
    })

    # Arrange columns in the exact order used during training
    input_data = input_data[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(
        f"🚗 Predicted Number of Vehicles: {prediction[0]:.2f}"
    )