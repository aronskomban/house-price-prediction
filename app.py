import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load the trained model
# -------------------------------
model = pickle.load(open('house_price_model.pkl', 'rb'))

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="🏠 UAE House Price Prediction", layout="centered")
st.title("🏠 UAE House Price Prediction App")
st.write("Predict house prices based on property features like size, rooms, and furnishing.")

# -------------------------------
# User Input Section
# -------------------------------
st.header("Enter Property Details")

# Numeric Inputs
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)
sizeMin = st.number_input("Property Size (sqft)", min_value=100, max_value=20000, value=1200)

# Select Inputs
type_ = st.selectbox("Property Type", ["Apartment", "Villa", "Townhouse"])
furnishing = st.selectbox("Furnishing", ["Furnished", "Unfurnished", "Partly Furnished"])
displayAddress = st.text_input("Location (Area/Community)", "Dubai Marina")

# -------------------------------
# Feature Engineering (as per training)
# -------------------------------

# Create a DataFrame for model input
input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sizeMin': [sizeMin],
    'type': [type_],
    'furnishing': [furnishing],
    'displayAddress': [displayAddress]
})

# -------------------------------
# One-Hot Encode Categorical Features
# -------------------------------
columns_needed = [
    'bedrooms', 'bathrooms', 'sizeMin',
    'type_Apartment', 'type_Villa', 'type_Townhouse',
    'furnishing_Furnished', 'furnishing_Unfurnished', 'furnishing_Partly Furnished'
]

# Encode input manually
for col in ['type_Apartment', 'type_Villa', 'type_Townhouse']:
    input_data[col] = 1 if col.split('_')[1] == type_ else 0

for col in ['furnishing_Furnished', 'furnishing_Unfurnished', 'furnishing_Partly Furnished']:
    input_data[col] = 1 if col.split('_')[1] == furnishing else 0

# Drop unused columns
input_data = input_data.drop(['type', 'furnishing', 'displayAddress'], axis=1, errors='ignore')

# Ensure correct order
for col in columns_needed:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[columns_needed]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: AED {prediction:,.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Trained on UAE Real Estate Dataset (Kaggle)")
