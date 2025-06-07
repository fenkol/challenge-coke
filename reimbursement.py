import pandas as pd
import numpy as np
import joblib

class ReimbursementCalculator:
    def __init__(self, model_path='reimbursement_model.joblib'):
        self.model = joblib.load(model_path)

    def calculate(self, trip_duration_days, miles_traveled, total_receipts_amount):
        # Feature engineering must be consistent with training
        miles_per_day = miles_traveled / trip_duration_days if trip_duration_days > 0 else 0
        receipts_per_day = total_receipts_amount / trip_duration_days if trip_duration_days > 0 else 0

        # Create a DataFrame for the model
        input_data = pd.DataFrame({
            'trip_duration_days': [trip_duration_days],
            'miles_traveled': [miles_traveled],
            'total_receipts_amount': [total_receipts_amount],
            'miles_per_day': [miles_per_day],
            'receipts_per_day': [receipts_per_day]
        })

        prediction = self.model.predict(input_data)
        return round(prediction[0], 2)

if __name__ == '__main__':
    # This part is for testing the calculator with public_cases.json
    import json

    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)

    calculator = ReimbursementCalculator()
    
    results = []
    for case in test_cases:
        inputs = case['input']
        expected = case['expected_output']
        
        calculated = calculator.calculate(
            inputs['trip_duration_days'],
            inputs['miles_traveled'],
            inputs['total_receipts_amount']
        )
        
        results.append({
            'trip_duration_days': inputs['trip_duration_days'],
            'miles_traveled': inputs['miles_traveled'],
            'total_receipts_amount': inputs['total_receipts_amount'],
            'expected': expected,
            'calculated': calculated,
            'difference': calculated - expected
        })

    results_df = pd.DataFrame(results)
    print("Test Results:")
    print(results_df.head())
    
    mean_abs_error = np.mean(np.abs(results_df['difference']))
    print(f"\nMean Absolute Error: {mean_abs_error}")

    # Further analysis of errors
    print("\nError distribution:")
    print(results_df['difference'].describe()) 