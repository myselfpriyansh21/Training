import streamlit as st
import pandas as pd
import joblib

# Load pipeline components
hpp = joblib.load("house_rent_prediction.pkl")

encoder = hpp["encoder"]
scaler = hpp["scaler"]
model = hpp["model"]

st.set_page_config(page_title="House Rent Prediction", page_icon="🏠")

st.title("🏠 House Rent Prediction")
st.write("Enter house details to predict monthly rent.")

# User Inputs
bhk = st.number_input("BHK", min_value=1, max_value=20, value=2)

size = st.number_input(
    "Size (sq.ft)",
    min_value=100,
    max_value=10000,
    value=1200
)

area_type = st.selectbox(
    "Area Type",
    ["Super Area", "Carpet Area", "Built Area"]
)

city = st.text_input("City", "Delhi")

furnishing = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

tenant = st.selectbox(
    "Tenant Preferred",
    ["Bachelors", "Family", "Bachelors/Family"]
)

bathroom = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=20,
    value=2
)

contact = st.selectbox(
    "Point of Contact",
    ["Contact Owner", "Contact Agent", "Contact Builder"]
)

# Prediction
if st.button("Predict Rent"):

    new_house = pd.DataFrame({
        "BHK": [bhk],
        "Size": [size],
        "Area Type": [area_type],
        "City": [city],
        "Furnishing Status": [furnishing],
        "Tenant Preferred": [tenant],
        "Bathroom": [bathroom],
        "Point of Contact": [contact]
    })

    try:
        encoded = encoder.transform(new_house)
        scaled = scaler.transform(encoded)

        prediction = model.predict(scaled)[0]

        st.success(
            f"Predicted Monthly Rent: ₹ {prediction:,.0f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")