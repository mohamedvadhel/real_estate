import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data():
    """
    Function to load the dataset from the CSV file.
    
    Returns:
        pandas.DataFrame: The loaded DataFrame containing the dataset.
    """
    data = pd.read_csv("housing_dataset.csv")
    return data

def train_model(data):
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
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Save the trained model to a file
    joblib.dump(model, 'model.pkl')

    return model

if __name__ == '__main__':
    data = load_data()
    model = train_model(data)
