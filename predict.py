import sys
import pandas as pd
import joblib

def predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    model = joblib.load('reimbursement_model.joblib')

    # Feature engineering
    miles_per_day = miles_traveled / trip_duration_days if trip_duration_days > 0 else 0
    receipts_per_day = total_receipts_amount / trip_duration_days if trip_duration_days > 0 else 0

    input_data = pd.DataFrame({
        'trip_duration_days': [trip_duration_days],
        'miles_traveled': [miles_traveled],
        'total_receipts_amount': [total_receipts_amount],
        'miles_per_day': [miles_per_day],
        'receipts_per_day': [receipts_per_day]
    })

    prediction = model.predict(input_data)
    print(round(prediction[0], 2))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)

    trip_duration_days = int(sys.argv[1])
    miles_traveled = float(sys.argv[2])
    total_receipts_amount = float(sys.argv[3])

    predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount) 