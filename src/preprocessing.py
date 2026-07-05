import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Charge le dataset.
    """
    df = pd.read_csv(file_path)
    return df


def feature_engineering(df):
    """
    Création de nouvelles variables métier.
    """

    # Différence de solde côté émetteur
    df["balance_diff_orig"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    # Différence de solde côté destinataire
    df["balance_diff_dest"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    return df


def encode_features(df):
    """
    Encodage de la variable catégorielle 'type'.
    """

    df = pd.get_dummies(
        df,
        columns=["type"],
        drop_first=True,
        dtype=int
    )

    return df


def remove_unused_columns(df):
    """
    Suppression des identifiants de comptes.
    """

    columns_to_drop = ["nameOrig", "nameDest"]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    return df


def split_features_target(df):
    """
    Séparation X et y.
    """

    y = df["isFraud"]

    X = df.drop(
        columns=["isFraud"]
    )

    return X, y


def train_test_split_data(X, y):
    """
    Séparation train/test.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(file_path):
    """
    Pipeline complet de prétraitement.
    """

    print("Chargement des données...")

    df = load_data(file_path)

    print(f"Shape initiale : {df.shape}")

    df = feature_engineering(df)

    df = encode_features(df)

    df = remove_unused_columns(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    FILE_PATH = "data/creditcard.csv"

    X_train, X_test, y_train, y_test = preprocess_data(
        FILE_PATH
    )

    print("\n✅ Prétraitement terminé avec succès.")