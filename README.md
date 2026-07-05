# 💳 FRAUD DETECTION BANKING

Une application complète de détection de fraude bancaire basée sur le Machine Learning, permettant d'identifier automatiquement les transactions frauduleuses à partir des données financières issues du dataset PaySim.

---

## Aperçu du projet

Ce projet couvre l'ensemble du cycle de vie d'un projet Data Science :

- Exploration et analyse des données (EDA)
- Prétraitement des données
- Feature Engineering
- Entraînement de modèles Machine Learning
- Comparaison de modèles
- Sélection du meilleur modèle
- Sauvegarde du modèle
- API REST avec Flask
- Dashboard interactif avec Streamlit

---

## Objectif

Détecter si une transaction bancaire est :

- ✅ Légitime
- 🚨 Frauduleuse

à partir des informations financières liées à la transaction.

---

## Dataset

Dataset utilisé :

## PaySim

Variables principales :

- step
- type
- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest
- isFraud
- isFlaggedFraud

Taille du dataset :

- 6 362 620 transactions
- 11 variables initiales

---

## Technologies utilisées

### Langage

- Python 3.12

### Analyse des données

- Pandas
- NumPy

### Visualisation

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- XGBoost

### Déploiement

- Flask
- Streamlit

### Sauvegarde du modèle

- Joblib

---
EDA.ipynb

pip install -r requirements.txt

python src/preprocessing.py

python src/train.py

python src/predict.py

python app/app.py

streamlit run app/streamlit_app.py


## 📁 Structure du projet

```text
Fraud-Detection/
│
├── app/
│   ├── app.py
│   └── streamlit_app.py
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_model.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
│
└── README.md

```