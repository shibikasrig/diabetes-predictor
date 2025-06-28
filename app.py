from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("diabetes_model.pkl")  # make sure this file exists!

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        risk = ""
        if probability < 0.3:
            risk = "Low Risk"
        elif probability < 0.7:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"

        return f"""
        <h2>🩺 Diabetes Risk Result</h2>
        <p><b>Prediction:</b> {"Diabetic" if prediction == 1 else "Non-Diabetic"}</p>
        <p><b>Probability:</b> {probability:.2f}</p>
        <p><b>Risk Level:</b> {risk}</p>
        <br><a href="/">Back to Home</a>
        """

    except Exception as e:
        print("Error:", e)
        return "❌ Error in prediction. Please check inputs."

if __name__ == '__main__':
    app.run(debug=True)
