from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = [
        float(request.form["male"]),
        float(request.form["age"]),
        float(request.form["education"]),
        float(request.form["currentSmoker"]),
        float(request.form["cigsPerDay"]),
        float(request.form["BPMeds"]),
        float(request.form["prevalentStroke"]),
        float(request.form["prevalentHyp"]),
        float(request.form["diabetes"]),
        float(request.form["totChol"]),
        float(request.form["sysBP"]),
        float(request.form["diaBP"]),
        float(request.form["BMI"]),
        float(request.form["heartRate"]),
        float(request.form["glucose"])
    ]

    data = np.array(data).reshape(1, -1)

    # Scale the input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)[0]

    # Probability
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction == 1:
        result = "High Risk"
    else:
        result = "Low Risk"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)