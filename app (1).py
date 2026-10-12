import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

st.set_page_config(
    page_title="University Admission Cut-off Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 University Admission Cut-off Predictor")
st.write("Predict the estimated admission cut-off score based on historical data using our Tuned Random Forest Model.")

# Input fields matching model schema
university_options = [
    'Banaras Hindu University', 'University of Delhi', 
    'Guru Gobind Singh Indraprastha University', 
    'Indian Council of Agricultural Research (ICAR)', 'University of Kashmir'
]
round_options = ['1', '2', '3', 'spot', 'none']
category_options = ['UR', 'OBC-NCL', 'SC', 'ST', 'EWS', 'PwBD']
seat_options = ['general', 'Paid Fee Course', 'Delhi region', 'Outside Delhi']
basis_options = ['official', 'merit-list']

# Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        university = st.selectbox("University", options=university_options)
        year = st.number_input("Admission Year", min_value=2024, max_value=2030, value=2026, step=1)
        round_val = st.selectbox("Admission Round", options=round_options)
        
    with col2:
        category = st.selectbox("Reservation Category", options=category_options)
        seat_type = st.selectbox("Seat Type", options=seat_options)
        basis = st.selectbox("Admission Basis", options=basis_options)
        
    submit_button = st.form_submit_button(label="Predict Cut-off")

if submit_button:
    # Create a DataFrame for model input matching feature schema
    input_data = pd.DataFrame([{
        'university': university,
        'year': year,
        'round': round_val,
        'category': category,
        'seat_type': seat_type,
        'basis': basis
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Estimated Cut-off Score: **{prediction:.2f}**")
    st.info("Note: Cut-off scores are capped at the historical 95th percentile (approx. 3007) for variance stability.")
