import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')


st.set_page_config(
    page_title='Marks Predictor',
    page_icon='🥩',
    layout='centered'
)

st.title('🐮 Students Marks Predictor')
st.write(
    'Enter the number of hours studied 🐓 (1 to 10) and click Predict to see the predicted marks'
)

# --- Load the model safely ---
def load_model(model_file):
    if not os.path.exists(model_file):
        st.error(f"Pickle file '{model_file}' not found in the current folder!")
        st.stop()
    with open(model_file, 'rb') as file:
        return pickle.load(file)

model = load_model('simple_lin_reg.pkl')

# --- Number input for hours ---
hours = st.number_input(
    'Hours_Studied',
    min_value=1.0,
    max_value=10.0,
    value=4.0,
    step=0.1,
    format='%.1f'
)

# --- Predict button ---
if st.button('Predict'):
    try:
        X = np.array([[hours]])
        predictions = model.predict(X)[0]
        st.success(f'Predicted marks: {predictions:.2f}')
        st.write('Note: This is a machine learning prediction. **Result may vary.**')
    except Exception as e:
        st.error(f'Prediction failed: {e}')
        st.exception(e)
