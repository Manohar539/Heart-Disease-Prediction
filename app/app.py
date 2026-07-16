from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request


# ==========================================================
# Flask Application
# ==========================================================

app = Flask(__name__)


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ==========================================================
# Load the Three Deployment Models
# ==========================================================

framingham_model = joblib.load(
    MODEL_DIR / "framingham_model.pkl"
)

heart_model = joblib.load(
    MODEL_DIR / "heart_model.pkl"
)

heart_failure_model = joblib.load(
    MODEL_DIR / "heart_failure_model.pkl"
)

print(
    "Framingham Model:",
    type(framingham_model).__name__
)

print(
    "Heart Model:",
    type(heart_model).__name__
)

print(
    "Heart Failure Model:",
    type(heart_failure_model).__name__
)


# ==========================================================
# Page Routes
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/heart")
def heart_page():
    return render_template("heart.html")


@app.route("/heart-failure")
def heart_failure_page():
    return render_template("heart_failure.html")


@app.route("/comparison")
def comparison_page():
    return render_template("comparison.html")


@app.route("/about")
def about():
    return render_template("about.html")


# ==========================================================
# Framingham Prediction Route
# ==========================================================

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

        patient_data = np.array(
            patient_data,
            dtype=float
        ).reshape(1, -1)

        probability = float(
            framingham_model.predict_proba(
                patient_data
            )[0][1]
        )

        if probability >= 0.70:
            risk = "High Risk"
            color = "red"
            recommendation = (
                "Consult a cardiologist and monitor blood pressure, "
                "cholesterol, glucose, diet, exercise and smoking."
            )

        elif probability >= 0.40:
            risk = "Moderate Risk"
            color = "yellow"
            recommendation = (
                "Schedule a medical check-up and improve lifestyle, "
                "diet and exercise."
            )

        else:
            risk = "Low Risk"
            color = "green"
            recommendation = (
                "Maintain regular exercise, balanced nutrition, "
                "routine check-ups and avoid smoking."
            )

        return render_template(
            "result.html",
            prediction=risk,
            probability=round(
                probability * 100,
                2
            ),
            color=color,
            recommendation=recommendation,
            age=request.form["age"],
            gender=(
                "Male"
                if request.form["male"] == "1"
                else "Female"
            ),
            smoker=(
                "Yes"
                if request.form["currentSmoker"] == "1"
                else "No"
            ),
            bmi=request.form["BMI"],
            cholesterol=request.form["totChol"],
            systolic=request.form["sysBP"],
            diastolic=request.form["diaBP"],
            glucose=request.form["glucose"]
        )

    except (KeyError, TypeError, ValueError) as error:
        return (
            f"<h2>Prediction Input Error</h2>"
            f"<p>{error}</p>",
            400
        )

    except Exception as error:
        app.logger.exception(
            "Framingham prediction failed."
        )

        return (
            f"<h2>Prediction Error</h2>"
            f"<p>{error}</p>",
            500
        )


# ==========================================================
# Heart Dataset Prediction Route
# ==========================================================

@app.route("/predict-heart", methods=["POST"])
def predict_heart():
    try:
        heart_data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["dataset"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalch"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"])
        ]

        heart_data = np.array(
            heart_data,
            dtype=float
        ).reshape(1, -1)

        probability = float(
            heart_model.predict_proba(
                heart_data
            )[0][1]
        )

        if probability >= 0.70:
            risk = (
                "Heart Disease Detected — "
                "High Probability"
            )
            color = "red"
            recommendation = (
                "Arrange a medical consultation promptly. "
                "The model indicates a high probability of "
                "heart disease based on the entered values."
            )

        elif probability >= 0.40:
            risk = (
                "Heart Disease Detected — "
                "Moderate Probability"
            )
            color = "yellow"
            recommendation = (
                "Consider scheduling a clinical assessment "
                "and reviewing blood pressure, cholesterol, "
                "symptoms and exercise response."
            )

        else:
            risk = (
                "Low Probability of Heart Disease"
            )
            color = "green"
            recommendation = (
                "Maintain a healthy lifestyle and continue "
                "routine medical check-ups."
            )

        return render_template(
            "result.html",
            prediction=risk,
            probability=round(
                probability * 100,
                2
            ),
            color=color,
            recommendation=recommendation,
            age=request.form["age"],
            gender=(
                "Male"
                if request.form["sex"] == "1"
                else "Female"
            ),
            smoker="Not included in this module",
            bmi="Not included",
            cholesterol=request.form["chol"],
            systolic=request.form["trestbps"],
            diastolic="Not included",
            glucose=(
                "Above 120 mg/dL"
                if request.form["fbs"] == "1"
                else "120 mg/dL or below"
            )
        )

    except (KeyError, TypeError, ValueError) as error:
        return (
            f"<h2>Heart Prediction Input Error</h2>"
            f"<p>{error}</p>",
            400
        )

    except Exception as error:
        app.logger.exception(
            "Heart dataset prediction failed."
        )

        return (
            f"<h2>Heart Prediction Error</h2>"
            f"<p>{error}</p>",
            500
        )


# ==========================================================
# Heart Failure Prediction Route
# ==========================================================

@app.route(
    "/predict-heart-failure",
    methods=["POST"]
)
def predict_heart_failure():
    try:
        failure_data = [
            float(request.form["age"]),
            float(request.form["anaemia"]),
            float(
                request.form[
                    "creatinine_phosphokinase"
                ]
            ),
            float(request.form["diabetes"]),
            float(
                request.form[
                    "ejection_fraction"
                ]
            ),
            float(
                request.form[
                    "high_blood_pressure"
                ]
            ),
            float(request.form["platelets"]),
            float(
                request.form[
                    "serum_creatinine"
                ]
            ),
            float(
                request.form[
                    "serum_sodium"
                ]
            ),
            float(request.form["sex"]),
            float(request.form["smoking"]),
            float(request.form["time"])
        ]

        failure_data = np.array(
            failure_data,
            dtype=float
        ).reshape(1, -1)

        probability = float(
            heart_failure_model.predict_proba(
                failure_data
            )[0][1]
        )

        if probability >= 0.70:
            risk = (
                "High Risk of Adverse "
                "Heart Failure Outcome"
            )
            color = "red"
            recommendation = (
                "Seek prompt medical assessment. "
                "The model indicates a high probability "
                "of an adverse heart failure outcome based "
                "on the entered values."
            )

        elif probability >= 0.40:
            risk = (
                "Moderate Risk of Adverse "
                "Heart Failure Outcome"
            )
            color = "yellow"
            recommendation = (
                "Schedule a clinical review and monitor "
                "ejection fraction, serum creatinine, "
                "serum sodium, blood pressure and other "
                "relevant indicators."
            )

        else:
            risk = (
                "Low Risk of Adverse "
                "Heart Failure Outcome"
            )
            color = "green"
            recommendation = (
                "Continue routine clinical monitoring and "
                "follow the treatment plan provided by a "
                "qualified healthcare professional."
            )

        return render_template(
            "result.html",
            prediction=risk,
            probability=round(
                probability * 100,
                2
            ),
            color=color,
            recommendation=recommendation,
            age=request.form["age"],
            gender=(
                "Male"
                if request.form["sex"] == "1"
                else "Female"
            ),
            smoker=(
                "Yes"
                if request.form["smoking"] == "1"
                else "No"
            ),
            bmi="Not included",
            cholesterol="Not included",
            systolic=(
                "High"
                if request.form[
                    "high_blood_pressure"
                ] == "1"
                else "Not high"
            ),
            diastolic="Not included",
            glucose=(
                "Diabetes present"
                if request.form["diabetes"] == "1"
                else "Diabetes not present"
            )
        )

    except (KeyError, TypeError, ValueError) as error:
        return (
            "<h2>Heart Failure Prediction "
            f"Input Error</h2><p>{error}</p>",
            400
        )

    except Exception as error:
        app.logger.exception(
            "Heart failure prediction failed."
        )

        return (
            "<h2>Heart Failure Prediction "
            f"Error</h2><p>{error}</p>",
            500
        )


# ==========================================================
# Run Flask Application
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)