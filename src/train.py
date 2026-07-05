import joblib
import pandas as pd

from preprocessing import preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


def evaluate_model(model, X_test, y_test, model_name):

    print(f"\n{'=' * 60}")
    print(f"Évaluation : {model_name}")
    print(f"{'=' * 60}")

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)

    print(f"\nROC-AUC Score : {auc:.4f}")

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix :")
    print(confusion_matrix(y_test, y_pred))

    return auc


def main():

    print("\n=== Prétraitement ===")

    X_train, X_test, y_train, y_test = preprocess_data(
        "data/creditcard.csv"
    )

    #################################################
    # RANDOM FOREST
    #################################################

    print("\n=== Entraînement Random Forest ===")

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_auc = evaluate_model(
        rf,
        X_test,
        y_test,
        "Random Forest"
    )

    #################################################
    # XGBOOST
    #################################################

    print("\n=== Entraînement XGBoost ===")

    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    xgb.fit(X_train, y_train)

    xgb_auc = evaluate_model(
        xgb,
        X_test,
        y_test,
        "XGBoost"
    )

    #################################################
    # COMPARAISON
    #################################################

    print("\n=== Comparaison Finale ===")

    results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost"
        ],
        "ROC_AUC": [
            rf_auc,
            xgb_auc
        ]
    })

    print(results)

    #################################################
    # MEILLEUR MODÈLE
    #################################################

    if xgb_auc > rf_auc:
        best_model = xgb
        best_name = "XGBoost"
    else:
        best_model = rf
        best_name = "Random Forest"

    print(f"\n✅ Meilleur modèle : {best_name}")

    #################################################
    # IMPORTANCE DES VARIABLES
    #################################################

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n=== Top 15 Features ===")
    print(importance.head(15))

    #################################################
    # SAUVEGARDE
    #################################################

    joblib.dump(
        best_model,
        "models/fraud_model.pkl"
    )

    print(
        "\n✅ Modèle sauvegardé dans models/fraud_model.pkl"
    )


if __name__ == "__main__":
    main()