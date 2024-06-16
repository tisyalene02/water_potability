import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model, scaler, and RFE
classifier = joblib.load('water.pkl')
scaler = joblib.load('scaler.pkl')
rfe = joblib.load('rfe.pkl')

# Function to make predictions
def predict_potability(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    selected_data = scaled_data[:, rfe.support_]
    prediction = classifier.predict(selected_data)
    return 'Potable' if prediction == 1 else 'Not Potable'

# Function to load and display the dataset
def load_dataset():
    df = pd.read_csv('water_potability.csv')  # Adjust path based on your dataset location
    return df

# Streamlit interface for prediction page
def prediction_page():
    st.title('Water Potability Prediction')

    st.write("""
    This page allows you to predict the potability of water based on selected features.
    """)

    # Define the input sliders
    ph = st.slider('pH', min_value=0.0, max_value=14.0, step=0.1, value=7.0)
    hardness = st.slider('Hardness', min_value=0.0, max_value=500.0, step=0.1, value=150.0)
    solids = st.slider('Solids', min_value=0.0, max_value=50000.0, step=1.0, value=20000.0)
    chloramines = st.slider('Chloramines', min_value=0.0, max_value=20.0, step=0.1, value=7.0)
    sulfate = st.slider('Sulfate', min_value=0.0, max_value=500.0, step=0.1, value=200.0)
    conductivity = st.slider('Conductivity', min_value=0.0, max_value=1000.0, step=0.1, value=400.0)
    organic_carbon = st.slider('Organic Carbon', min_value=0.0, max_value=30.0, step=0.1, value=10.0)
    trihalomethanes = st.slider('Trihalomethanes', min_value=0.0, max_value=200.0, step=0.1, value=80.0)
    turbidity = st.slider('Turbidity', min_value=0.0, max_value=10.0, step=0.1, value=3.0)

    # Collect the input data into a list
    input_data = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]

    # Predict button
    if st.button('Predict Potability'):
        result = predict_potability(input_data)
        st.write(f'The water is {result}')

# Function to display dataset exploration
def dataset_page():
    st.title('Explore Dataset')

    st.write("""
    This page allows you to explore the dataset used for training the potability prediction model.
    """)

    df = load_dataset()

    # Display summary statistics
    st.header('Summary Statistics')
    st.write(df.describe())

    # Display histograms for numerical columns
    st.header('Histograms')
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    selected_column = st.selectbox('Select a column for histogram', numerical_columns)
    st.histogram(df[selected_column])

    # Display the dataset in a table
    st.header('Raw Data')
    st.dataframe(df)

# Main Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Prediction', 'Explore Dataset'])

    if page == 'Prediction':
        prediction_page()
    elif page == 'Explore Dataset':
        dataset_page()

if __name__ == '__main__':
    main()

