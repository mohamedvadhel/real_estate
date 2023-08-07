import streamlit as st
import pandas as pd
import joblib

def load_data():
    """
    Function to load the dataset from the CSV file.
    
    Returns:
        pandas.DataFrame: The loaded DataFrame containing the dataset.
    """
    data = pd.read_csv("housing_dataset.csv")
    return data

def load_model():
    # Load the trained model
    model = joblib.load('model.pkl')
    return model

def main():
    st.title("Housing Price Prediction in Mauritania")

    # Load the dataset
    data = load_data()

    # Load the trained model
    model = load_model()

    # Prediction for each feature
    st.subheader("Housing Features")

    features = {}  # Dictionary to hold feature values entered by the user

    for feature in data.columns.drop('Price'):
        features[feature] = st.number_input(feature, min_value=float(data[feature].min()), max_value=float(data[feature].max()), value=float(data[feature].mean()))

    # Make predictions
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)

    st.write(f"Predicted Housing Price: {prediction[0]:.2f}")

if __name__ == '__main__':
    main()
