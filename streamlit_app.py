import streamlit as st
import pandas as pd
import joblib

# Load the dataset for exploration
df = pd.read_csv('water_potability.csv')  # Adjust the path based on your dataset location

# Load the trained model, scaler, and RFE (from your previous code)
classifier = joblib.load('water.pkl')
scaler = joblib.load('scaler.pkl')
rfe = joblib.load('rfe.pkl')

# Function to make predictions (from your previous code)
def predict_potability(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    selected_data = scaled_data[:, rfe.support_]
    prediction = classifier.predict(selected_data)
    return 'Potable' if prediction == 1 else 'Not Potable'

# Streamlit interface
st.title('Water Potability Prediction')

st.write("""
This is a simple web application to predict the potability of water based on selected features.
""")

# Define the input sliders (from your previous code)
ph = st.slider('pH', min_value=0.0, max_value=14.0, step=0.1, value=7.0)
hardness = st.slider('Hardness', min_value=0.0, max_value=500.0, step=0.1, value=150.0)
solids = st.slider('Solids', min_value=0.0, max_value=50000.0, step=1.0, value=20000.0)
chloramines = st.slider('Chloramines', min_value=0.0, max_value=20.0, step=0.1, value=7.0)
sulfate = st.slider('Sulfate', min_value=0.0, max_value=500.0, step=0.1, value=200.0)
conductivity = st.slider('Conductivity', min_value=0.0, max_value=1000.0, step=0.1, value=400.0)
organic_carbon = st.slider('Organic Carbon', min_value=0.0, max_value=30.0, step=0.1, value=10.0)
trihalomethanes = st.slider('Trihalomethanes', min_value=0.0, max_value=200.0, step=0.1, value=80.0)
turbidity = st.slider('Turbidity', min_value=0.0, max_value=10.0, step=0.1, value=3.0)

# Collect the input data into a list (from your previous code)
input_data = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]

# Predict button (from your previous code)
if st.button('Predict Potability'):
    result = predict_potability(input_data)
    st.write(f'The water is {result}')

# Section for exploring the dataset
st.sidebar.header('Explore Dataset')

# Display the dataset in a collapsible sidebar section
with st.sidebar.expander("View Dataset", expanded=True):
    st.dataframe(df)

# Optionally, add more interactive features to explore the dataset (sorting, filtering, etc.)
