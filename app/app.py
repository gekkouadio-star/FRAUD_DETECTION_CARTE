from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/fraud_model.pkl"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        "Le modèle fraud_model.pkl est introuvable."
    )

model = joblib.load(MODEL_PATH)

# ==========================================
# ROUTE ACCUEIL
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "application": "Fraud Detection API",
        "version": "1.0",
        "status": "running"
    })

# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    })

# ==========================================
# PRÉDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Aucune donnée reçue."
            }), 400

        transaction = pd.DataFrame([data])

        prediction = int(
            model.predict(transaction)[0]
        )

        probability = float(
            model.predict_proba(transaction)[0][1]
        )

        if probability < 0.30:
            risk_level = "LOW"

        elif probability < 0.70:
            risk_level = "MEDIUM"

        else:
            risk_level = "HIGH"

        response = {
            "prediction": prediction,
            "status":
                "FRAUD"
                if prediction == 1
                else "LEGITIMATE",

            "probability": round(
                probability,
                4
            ),

            "risk_level": risk_level,

            "analysis_time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# ERREUR 404
# ==========================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Route introuvable"
    }), 404

# ==========================================
# LANCEMENT
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )