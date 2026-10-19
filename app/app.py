import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.predict import load_model, get_neighborhoods, predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.caption("Estimasi harga rumah Ames, Iowa (USD) · Random Forest · R² ≈ 0.81")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
neighborhoods = get_neighborhoods(model)

with st.form("predict_form"):
    location = st.selectbox("📍 Location", neighborhoods)
    size = st.slider("🏠 Size (sqft)", 300, 6000, 1500, step=50)
    bedrooms = st.number_input("🛏️ Bedrooms", 0, 10, 3)
    bathrooms = st.number_input("🛁 Bathrooms", 0.0, 6.0, 2.0, step=0.5)
    parking = st.number_input("🚗 Parking (cars)", 0, 4, 2)
    submitted = st.form_submit_button("Predict price")

if submitted:
    price = predict_price(model, location, size, bedrooms, bathrooms, parking)
    st.success(f"Estimated price: **${price:,.0f}**")