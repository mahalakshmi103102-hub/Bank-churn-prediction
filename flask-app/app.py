from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
with open('churn_stacked_model.pkl', 'rb') as model_file:
    stacked_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Expected feature names
feature_names = [
    'credit_score', 'gender', 'age', 'tenure', 'balance', 
    'products_number', 'credit_card', 'active_member', 
    'estimated_salary', 'country_Germany', 'country_Spain'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Encode categorical variables
        gender_encoded = 1 if data['gender'].lower() == "male" else 0
        country_Germany = 1 if data['country'] == "Germany" else 0
        country_Spain = 1 if data['country'] == "Spain" else 0

        # Convert input values to correct format
        input_data = pd.DataFrame([[
            float(data['credit_score']), gender_encoded, float(data['age']),
            int(data['tenure']), float(data['balance']), int(data['products_number']),
            int(data['credit_card']), int(data['active_member']), float(data['estimated_salary']),
            country_Germany, country_Spain
        ]], columns=feature_names)

        # Scale input data
        input_data_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = stacked_model.predict(input_data_scaled)[0]
        result = "Churn" if prediction == 1 else "Not Churn"

    except Exception as e:
        result = f"❌ Error in prediction: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
