import joblib
import pandas as pd


def predict_transaction():

    print("Chargement du modèle...")

    model = joblib.load(
        "models/fraud_model.pkl"
    )

    transaction = {
        "step": 1,
        "amount": 10000,
        "oldbalanceOrg": 20000,
        "newbalanceOrig": 10000,
        "oldbalanceDest": 0,
        "newbalanceDest": 10000,
        "isFlaggedFraud": 0,
        "balance_diff_orig": 10000,
        "balance_diff_dest": 10000,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 1
    }

    data = pd.DataFrame([transaction])

    print("\nTransaction testée :")
    print(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    print("\n===== RÉSULTAT =====")

    if prediction == 1:
        print("🚨 FRAUDE DÉTECTÉE")
    else:
        print("✅ TRANSACTION LÉGITIME")

    print(
        f"Probabilité de fraude : {probability:.4f}"
    )


if __name__ == "__main__":
    predict_transaction()