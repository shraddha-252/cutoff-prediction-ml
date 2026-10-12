import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- 1. Train (or load) the model ---
# Cached so this only runs once per app instance, not on every interaction.
# Training directly from the CSV on startup avoids relying on .pkl files being
# committed to the repo (and avoids sklearn-version pickle mismatches between
# your local machine and the deployment environment).
@st.cache_resource
def load_components():
    df = pd.read_csv("Delivery_Logistics.csv")

    X = df.drop(columns=[
        'delivery_id',
        'delivery_time_hours',
        'expected_time_hours',
        'delayed',
        'delivery_status'  # would leak the target
    ])
    y = df['delayed']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train_columns = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # KNN is distance-based, so scale features (delivery_cost and delivery_rating
    # are on very different scales and would otherwise dominate the distance calc)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    return model, le, scaler, X_train_columns

knn_model, le, scaler, X_train_columns = load_components()

# --- 2. Define Prediction Function ---
def predict_delayed(input_data, X_train_columns, le, scaler, model):
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_df_final = input_df_encoded.reindex(columns=X_train_columns, fill_value=0)
    input_scaled = scaler.transform(input_df_final)
    prediction_numeric = model.predict(input_scaled)
    prediction_label = le.inverse_transform(prediction_numeric)
    return prediction_label[0]

# --- 3. Streamlit UI ---
st.title('KNN Delivery Delay Prediction App')
st.write('Enter the details of the delivery to predict if it will be delayed.')

distance_km = st.number_input('Distance (km)', min_value=0.1, max_value=300.0, value=150.0)
package_weight_kg = st.number_input('Package Weight (kg)', min_value=0.1, max_value=50.0, value=25.0)
delivery_rating = st.slider('Delivery Rating', min_value=1, max_value=5, value=4)
delivery_cost = st.number_input('Delivery Cost', min_value=0.0, max_value=2000.0, value=800.0)

# These match the exact category values present in Delivery_Logistics.csv
delivery_partner_options = ['amazon logistics', 'blue dart', 'delhivery', 'dhl', 'ecom express', 'ekart', 'fedex', 'shadowfax', 'xpressbees']
package_type_options = ['automobile parts', 'clothing', 'cosmetics', 'documents', 'electronics', 'fragile items', 'furniture', 'groceries', 'pharmacy']
vehicle_type_options = ['bike', 'ev bike', 'ev van', 'scooter', 'truck', 'van']
delivery_mode_options = ['express', 'same day', 'standard', 'two day']
region_options = ['central', 'east', 'north', 'south', 'west']
weather_condition_options = ['clear', 'cold', 'foggy', 'hot', 'rainy', 'stormy']

delivery_partner = st.selectbox('Delivery Partner', options=delivery_partner_options)
package_type = st.selectbox('Package Type', options=package_type_options)
vehicle_type = st.selectbox('Vehicle Type', options=vehicle_type_options)
delivery_mode = st.selectbox('Delivery Mode', options=delivery_mode_options)
region = st.selectbox('Region', options=region_options)
weather_condition = st.selectbox('Weather Condition', options=weather_condition_options)

input_features = {
    'distance_km': distance_km,
    'package_weight_kg': package_weight_kg,
    'delivery_rating': delivery_rating,
    'delivery_cost': delivery_cost,
    'delivery_partner': delivery_partner,
    'package_type': package_type,
    'vehicle_type': vehicle_type,
    'delivery_mode': delivery_mode,
    'region': region,
    'weather_condition': weather_condition
}

if st.button('Predict Delay'):
    prediction = predict_delayed(input_features, X_train_columns, le, scaler, knn_model)
    if prediction == 'yes':
        st.error('Prediction: The delivery is likely to be **Delayed**')
    else:
        st.success('Prediction: The delivery is likely to be **On Time**')

# To run: place Delivery_Logistics.csv in the same folder as this file, then
# `streamlit run app.py`. No .pkl files needed — the model trains on startup.
