import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression


# Load trained pipeline
pipe = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# Load OneHotEncoder if it was saved separately
try:
    encoder = pickle.load(open("OneHotEncoder.pkl", "rb"))
except FileNotFoundError:
    encoder = None  # If encoder is in the pipeline, we don't need a separate file

st.title("Car Price Prediction App🚕")

# Input fields

company_models = {
    "Hyundai": ["Hyundai Santro Xing", "Hyundai Grand i10", "Hyundai Eon", "Hyundai Elite i20", "Hyundai Creta"],
    "Mahindra": ["Mahindra Jeep CL550", "Mahindra Scorpio SLE", "Mahindra XUV500 W8", "Mahindra Thar CRDe"],
    "Ford": ["Ford EcoSport Titanium", "Ford Figo", "Ford EcoSport Ambiente", "Ford Endeavor 4x4"],
    "Maruti": ["Maruti Suzuki Alto", "Maruti Suzuki Swift", "Maruti Suzuki Wagon R", "Maruti Suzuki Baleno"],
    "Toyota": ["Toyota Corolla Altis", "Toyota Fortuner", "Toyota Etios", "Toyota Innova 2.5","Toyota Fortuner 3.0","Toyota Corolla H2"],
    "Honda": ["Honda City", "Honda Amaze", "Honda Jazz", "Honda Accord",""],
    "Audi": ["Audi A8", "Audi Q7", "Audi A4 1.8", "Audi A6 2.0"],
    "BMW": ["BMW 3 Series", "BMW 5 Series", "BMW X1 xDrive20d"],
    "Mercedes": ["Mercedes Benz C", "Mercedes Benz A", "Mercedes Benz GLA"],
    "Volkswagen": ["Volkswagen Polo", "Volkswagen Jetta", "Volkswagen Passat Diesel"],
    "Jeep": ["Jeep Wrangler Unlimited", "Jeep Compass", "Jeep Grand Cherokee"]
}

# Company selection
company = st.selectbox("Brand Name", list(company_models.keys()))

# Dynamically update car models based on selected company
models = company_models.get(company, [])  # Get models for selected brand
# name = st.selectbox("Car Name (Brand Model)", models)



name = st.selectbox("Car Name (Brand Model)",models)

year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG"])
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
            predicted_price = abs(pipe.predict(input_data)[0])
            st.success(f"Estimated Price: PKR {predicted_price:,.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
