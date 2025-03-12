import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# Load trained pipeline
pipe = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# Load OneHotEncoder if it was saved separately
try:
    encoder = pickle.load(open("OneHotEncoder.pkl", "rb"))
except FileNotFoundError:
    encoder = None  # If encoder is in the pipeline, we don't need a separate file

st.title("Car Price Prediction App")

# Input fields
name = st.text_input("Car Name (Brand Model)")
company = st.text_input("Company Name")
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)

if st.button("Predict Price"):
    # Creating DataFrame with correct column names
    input_data = pd.DataFrame([[name, company, year, fuel_type, kms_driven]],
                              columns=['name', 'company', 'year', 'fuel_type', 'kms_driven'])

    # Ensure data types are correct
    input_data['year'] = input_data['year'].astype(int)
    input_data['kms_driven'] = input_data['kms_driven'].astype(int)

    # Ensure all necessary columns exist
    missing_cols = set(pipe.feature_names_in_) - set(input_data.columns)
    
    if missing_cols:
        st.error(f"Error in prediction: Missing columns: {missing_cols}")
    else:
        try:
            # Predict price
            predicted_price = pipe.predict(input_data)[0]
            st.success(f"Estimated Price: ₹ {predicted_price:,.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
