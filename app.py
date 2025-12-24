import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_california_model.pkl")

st.title("California House Price Prediction")
st.sidebar.header("Enter House Details")

def user_input_features():
    longitude = st.sidebar.number_input("Longitude", -124.35, -114.31, -119.0)
    latitude = st.sidebar.number_input("Latitude", 32.54, 41.95, 34.0)
    housing_median_age = st.sidebar.number_input("Housing Median Age", 1, 52, 20)
    total_rooms = st.sidebar.number_input("Total Rooms", 2, 10000, 1000)
    total_bedrooms = st.sidebar.number_input("Total Bedrooms", 1, 5000, 500)
    population = st.sidebar.number_input("Population", 1, 50000, 1000)
    households = st.sidebar.number_input("Households", 1, 5000, 300)
    median_income = st.sidebar.number_input("Median Income", 0.0, 15.0, 3.0)

    ocean_proximity = st.sidebar.selectbox(
        "Ocean Proximity",
        ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    )

    # 🔥 ENGINEERED FEATURES (IMPORTANT)
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    # Create dataframe with ALL required features
    input_df = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "rooms_per_household": [rooms_per_household],
        "bedrooms_per_room": [bedrooms_per_room],
        "population_per_household": [population_per_household],
        "ocean_proximity_INLAND": [0],
        "ocean_proximity_ISLAND": [0],
        "ocean_proximity_NEAR BAY": [0],
        "ocean_proximity_NEAR OCEAN": [0],
    })

    # One-hot encode ocean proximity
    input_df.loc[0, f"ocean_proximity_{ocean_proximity}"] = 1

    return input_df

input_df = user_input_features()
# Ensure the input dataframe has the same feature order as the model expects
input_df = input_df[model.feature_names_in_]

if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.subheader("Predicted House Price")
    st.write(f"${prediction[0]:,.2f}")
    st.write("Note: This is a predicted value and may not reflect the actual market price.")    