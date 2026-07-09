from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")
print("Loaded Model:", type(model).__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_data = [
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

        patient_data = np.array(patient_data).reshape(1, -1)

        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1]

        if probability >= 0.70:
            risk = "High Risk"
            color = "red"
            recommendation = "Consult a cardiologist and monitor BP, cholesterol, glucose, diet, and smoking."
        elif probability >= 0.40:
            risk = "Moderate Risk"
            color = "yellow"
            recommendation = "Schedule a medical check-up and improve lifestyle, diet, and exercise."
        else:
            risk = "Low Risk"
            color = "green"
            recommendation = "Maintain regular exercise, balanced diet, routine check-ups, and avoid smoking."

        return render_template(
            "result.html",
            prediction=risk,
            probability=round(probability * 100, 2),
            color=color,
            recommendation=recommendation
        )

    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p>"


if __name__ == "__main__":
    app.run(debug=True)