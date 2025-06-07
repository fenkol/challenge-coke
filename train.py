import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    inputs = [d['input'] for d in data]
    outputs = [d['expected_output'] for d in data]
    
    df = pd.DataFrame(inputs)
    df['expected_output'] = outputs
    
    return df

def feature_engineering(df):
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    # handle division by zero for trip_duration_days = 0, if any
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df

def train_model(df):
    features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'receipts_per_day']
    target = 'expected_output'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Gradient Boosting Regressor model...")
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error on test set: {mae}")

    print("Saving model to 'reimbursement_model.joblib'")
    joblib.dump(model, 'reimbursement_model.joblib')

if __name__ == '__main__':
    df = load_data('public_cases.json')
    df = feature_engineering(df)
    train_model(df) 