import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==========================
# CONFIGURATION DE LA PAGE
# ==========================

st.set_page_config(
    page_title="Fraud Detection Banking",
    page_icon="💳",
    layout="wide"
)

# ==========================
# CSS PERSONNALISÉ
# ==========================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.header-container {
    background: linear-gradient(
        90deg,
        #0f172a,
        #1e3a8a
    );

    border-radius: 20px;

    padding: 30px;

    border: 2px solid #3b82f6;

    text-align: center;

    margin-bottom: 20px;

    box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
}

.header-title {

    color: white;

    font-size: 48px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 2px;
}

.header-subtitle {

    color: #dbeafe;

    font-size: 18px;

    margin-top: 10px;
}

div[data-testid="metric-container"] {

    background-color: #f8fafc;

    border: 1px solid #e5e7eb;

    padding: 15px;

    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Machine Learning")

with col2:
    st.success("Modèle prêt")

with col3:
    st.warning("Analyse en temps réel")
# ==========================
# CHARGEMENT MODÈLE
# ==========================

MODEL_PATH = "models/fraud_model.pkl"

if not Path(MODEL_PATH).exists():
    st.error(
        "❌ Modèle introuvable.\n\nExécutez d'abord : python src/train.py"
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.title("💳 Fraud Detection")

    st.info(
        """
        ### Projet Machine Learning

        **Dataset :**
        PaySim

        **Modèles :**
        - Random Forest
        - XGBoost

        **Objectif :**
        Détecter les transactions frauduleuses.
        """
    )

    st.success("Modèle chargé")
    
    st.metric(
        "Dataset Size",
        "6.3M"
    )

    st.metric(
        "Features",
        "13"
    )

    st.metric(
        "Model Version",
        "1.0"
    )
    st.markdown("---")

    st.caption(
        "Projet réalisé par Gerard Kouadio"
    )

# ==========================
# HEADER
# ==========================

st.markdown(
    """
    <div class="header-container">
        <div class="header-title">
            💳 FRAUD DETECTION BANKING
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================
# KPIs GÉNÉRAUX
# ==========================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Transactions",
        "6.3M"
    )

with k2:
    st.metric(
        "Features",
        "13"
    )

with k3:
    st.metric(
        "Modèles",
        "2"
    )

with k4:
    st.metric(
        "Objectif",
        "Fraud Detection"
    )

st.divider()

# ==========================
# FORMULAIRE
# ==========================

left, right = st.columns(2)

with left:

    step = st.number_input(
        "⏱ Step",
        min_value=0,
        value=1
    )

    amount = st.number_input(
        "💰 Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalanceOrg = st.number_input(
        "🏦 Old Balance Origin",
        min_value=0.0,
        value=5000.0
    )

    newbalanceOrig = st.number_input(
        "🏦 New Balance Origin",
        min_value=0.0,
        value=4000.0
    )

with right:

    oldbalanceDest = st.number_input(
        "🏧 Old Balance Destination",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "🏧 New Balance Destination",
        min_value=0.0,
        value=1000.0
    )

    isFlaggedFraud = st.selectbox(
        "🚩 Flagged Fraud",
        [0, 1]
    )

    transaction_type = st.selectbox(
        "🔄 Transaction Type",
        [
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER"
        ]
    )

# ==========================
# PRÉDICTION
# ==========================

if st.button("Analyser la transaction"):

    balance_diff_orig = (
        oldbalanceOrg - newbalanceOrig
    )

    balance_diff_dest = (
        newbalanceDest - oldbalanceDest
    )

    transaction = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "type_CASH_OUT": 1 if transaction_type == "CASH_OUT" else 0,
        "type_DEBIT": 1 if transaction_type == "DEBIT" else 0,
        "type_PAYMENT": 1 if transaction_type == "PAYMENT" else 0,
        "type_TRANSFER": 1 if transaction_type == "TRANSFER" else 0
    }

    df = pd.DataFrame([transaction])

    prediction = model.predict(df)[0]

    probability = float(
        model.predict_proba(df)[0][1]
    )

    # ==========================
    # NIVEAU DE RISQUE
    # ==========================

    if probability < 0.30:
        risk_level = "🟢 LOW"

    elif probability < 0.70:
        risk_level = "🟡 MEDIUM"

    else:
        risk_level = "🔴 HIGH"

    st.divider()

    # ==========================
    # MÉTRIQUES
    # ==========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Amount",
            f"{amount:,.2f}"
        )

    with col3:
        st.metric(
            "Transaction Type",
            transaction_type
        )

    with col4:
        st.metric(
            "Risk Level",
            risk_level
        )

    # ==========================
    # BARRE DE RISQUE
    # ==========================

    st.subheader("Risk Analysis")

    st.progress(probability)

    confidence = (1 - abs(0.5 - probability) * 2) * 100

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )
    
    st.write(
        f"""
        **Montant :** {amount:,.2f}

        **Probabilité de fraude :** {probability:.2%}

        **Catégorie de risque :** {risk_level}
        """
    )

    # ==========================
    # RÉSULTAT
    # ==========================

    if prediction == 1:

        st.error(
            f"""
            🚨 FRAUD DETECTED

            Risk Level : {probability:.2%}
            """
        )

    else:

        st.success(
            f"""
            ✅ LEGITIMATE TRANSACTION

            Risk Level : {probability:.2%}
            """
        )
    # ==========================
    # RECOMMENDATION BUSINESS
    # ==========================
    st.subheader("Business Recommendation")

    if prediction == 1:

        st.warning(
            """
            Recommandation :

            • Suspendre temporairement la transaction

            • Vérifier l'identité du client

            • Déclencher une investigation de sécurité
            """
        )

    else:

        st.info(
            """
            Recommandation :

            • Transaction autorisée

            • Aucun comportement anormal détecté

            • Monitoring standard suffisant
            """
        )
    # ==========================
    # BALANCE INDICATORS
    # ==========================

    st.subheader("Balance Indicators")

    b1, b2 = st.columns(2)

    with b1:
        st.metric(
            "Balance Difference Origin",
            f"{balance_diff_orig:,.2f}"
        )

    with b2:
        st.metric(
            "Balance Difference Destination",
            f"{balance_diff_dest:,.2f}"
        )

    # ==========================
    # DATE D'ANALYSE
    # ==========================

    st.caption(
        f"🕒 Analyse effectuée le : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    # ==========================
    # DETAILS TRANSACTION
    # ==========================

    with st.expander("Transaction Details"):

        st.dataframe(
            df,
            use_container_width=True
        )

    # ==========================
    # TELECHARGEMENT CSV
    # ==========================

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Télécharger la transaction",
        data=csv,
        file_name="transaction.csv",
        mime="text/csv"
    )

# ==========================
# FOOTER
# ==========================

st.divider()

st.caption(
    """
    💳 FRAUD DETECTION BANKING

    Powered by Machine Learning

    Random Forest • XGBoost • Streamlit

    Developed by Gerard Kouadio
    """
)