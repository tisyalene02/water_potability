import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
def load_dataset():
    # Replace this with your dataset loading logic
    # For demonstration, let's create a sample DataFrame
    data = {
        'pH': np.random.uniform(0, 14, 100),
        'Hardness': np.random.uniform(0, 500, 100),
        'Solids': np.random.uniform(0, 50000, 100),
        'Chloramines': np.random.uniform(0, 20, 100),
        'Sulfate': np.random.uniform(0, 500, 100),
        'Conductivity': np.random.uniform(0, 1000, 100),
        'Organic Carbon': np.random.uniform(0, 30, 100),
        'Trihalomethanes': np.random.uniform(0, 200, 100),
        'Turbidity': np.random.uniform(0, 10, 100),
        'Potability': np.random.randint(0, 2, 100)  # Binary target for potability
    }
    df = pd.DataFrame(data)
    return df

# Function to make predictions
def predict_potability(input_data):
    # Replace with your model loading and prediction logic
    classifier = joblib.load('water.pkl')
    scaler = joblib.load('scaler.pkl')
    rfe = joblib.load('rfe.pkl')

    input_data = np.array(input_data).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    selected_data = scaled_data[:, rfe.support_]
    prediction = classifier.predict(selected_data)
    return prediction[0]

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

    # Display correlation matrix
    st.header('Correlation Matrix')
    corr_matrix = df.corr()
    st.write(corr_matrix)

    # Display the dataset in a table
    st.header('Raw Data')
    st.dataframe(df)

# Function to display prediction interface
def prediction_page():
    st.title('Water Potability Prediction')

    st.write("""
    This is a simple web application to predict the potability of water based on selected features.
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
        prediction = predict_potability(input_data)
        result = 'Potable' if prediction == 1 else 'Not Potable'
        st.write(f'The water is predicted to be {result}')

# Main Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Explore Dataset', 'Prediction'])

    if page == 'Explore Dataset':
        dataset_page()
    elif page == 'Prediction':
        prediction_page()

if __name__ == '__main__':
    main()
