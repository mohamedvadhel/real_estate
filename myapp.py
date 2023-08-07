import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    """
    Function to load the dataset from the CSV file.
    
    Returns:
        pandas.DataFrame: The loaded DataFrame containing the dataset.
    """
    data = pd.read_csv("housing_dataset.csv")
    return data

def main():
    st.title("Housing Price Prediction in Mauritania")

    # Load the dataset
    data = load_data()

    # Data exploration and preprocessing (if needed)
    # ...

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['Price'])
    y = data['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation (optional)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R2 Score: {r2}")

    # Prediction for each feature
    st.subheader("Housing Features")

    features = {}  # Dictionary to hold feature values entered by the user

    for feature in X.columns:
        features[feature] = st.number_input(feature, min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))

    # Make predictions
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)

    st.write(f"Predicted Housing Price: {prediction[0]:.2f}")

if __name__ == '__main__':
    main()
