from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'house_price_model.joblib'
METRICS_PATH = ROOT / 'models' / 'metrics.json'

st.set_page_config(page_title='House Price Prediction', page_icon='🏠', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    return json.loads(METRICS_PATH.read_text())

model = load_model()
meta = load_metadata()

st.title('🏠 House Price Prediction')
st.write('Enter the house characteristics below to estimate its price using a trained machine-learning regression model.')

with st.sidebar:
    st.header('Model Information')
    st.write(f"**Selected model:** {meta['best_model']}")
    st.write('The model was selected using the lowest test RMSE, with R² used as a tie-breaker.')
    st.divider()
    st.caption('Dataset: 3,000 generated house records for this project.')

col1, col2 = st.columns(2)
with col1:
    area = st.number_input('Area (sq ft)', min_value=500, max_value=4500, value=1500, step=50)
    bedrooms = st.number_input('Bedrooms', min_value=1, max_value=6, value=3, step=1)
    bathrooms = st.number_input('Bathrooms', min_value=1, max_value=5, value=2, step=1)
    stories = st.number_input('Stories', min_value=1, max_value=3, value=2, step=1)
    parking = st.number_input('Parking spaces', min_value=0, max_value=3, value=1, step=1)
    age = st.number_input('House age (years)', min_value=0, max_value=40, value=10, step=1)
with col2:
    distance = st.number_input('Distance to city center (km)', min_value=1.0, max_value=30.0, value=8.0, step=0.5)
    location_score = st.slider('Location score', 1.0, 10.0, 7.0, 0.1)
    condition_score = st.slider('Condition score', 1.0, 10.0, 7.0, 0.1)
    has_garden = st.selectbox('Garden', ['No', 'Yes'])
    has_basement = st.selectbox('Basement', ['No', 'Yes'])

if st.button('Predict House Price', type='primary', use_container_width=True):
    input_df = pd.DataFrame([{
        'area_sqft': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'age_years': age,
        'distance_to_city_km': distance,
        'location_score': location_score,
        'condition_score': condition_score,
        'has_garden': 1 if has_garden == 'Yes' else 0,
        'has_basement': 1 if has_basement == 'Yes' else 0,
    }])
    prediction = float(model.predict(input_df)[0])
    st.success(f'Estimated House Price: ₹{prediction:,.0f}')
    st.info('Interpretation: this is the model-estimated market value based on the characteristics entered. It is an estimate, not a professional property valuation.')

st.divider()
st.subheader('📊 Model Evaluation')
metrics = meta['metrics']
metric_df = pd.DataFrame(metrics).T.rename(columns={
    'MAE_INR': 'MAE (₹)', 'RMSE_INR': 'RMSE (₹)', 'R2': 'R²'
})
metric_df['MAE (₹)'] = metric_df['MAE (₹)'].map(lambda x: f'₹{x:,.0f}')
metric_df['RMSE (₹)'] = metric_df['RMSE (₹)'].map(lambda x: f'₹{x:,.0f}')
metric_df['R²'] = metric_df['R²'].map(lambda x: f'{x:.4f}')
st.dataframe(metric_df, use_container_width=True)

st.caption('For reproducibility, the training code uses random_state=42 and stores the selected model with joblib.')
