import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Toddler Autism dataset July 2018.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")


def load_data(path: str) -> pd.DataFrame:
    """Load the toddler autism dataset from CSV."""
    return pd.read_csv(path)


def build_pipeline(df: pd.DataFrame):
    """Build preprocessing + model pipeline using the same features as the web form.

    Features used:
    - A1–A10
    - Age_Mons
    - Sex, Ethnicity, Jaundice, Family_mem_with_ASD
    """

    # Target column name in the dataset
    target_col = "Class/ASD Traits "

    # Explicitly select only the features that are collected in the web form
    feature_cols = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "Age_Mons",
        "Sex",
        "Ethnicity",
        "Jaundice",
        "Family_mem_with_ASD",
    ]

    feature_df = df[feature_cols].copy()
    y = df[target_col].map({"Yes": 1, "No": 0}).values

    # Separate numeric and categorical columns explicitly
    numeric_cols = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "Age_Mons",
    ]
    categorical_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    return pipeline, feature_df, y


def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(DATA_PATH)

    pipeline, X, y = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # Evaluate metrics for logging purposes
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = pipeline.score(X_test, y_test)
    print(f"Validation accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        auc = None
        print("ROC AUC could not be computed (single class in y_test).")

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["No ASD Traits", "ASD Traits"],
        yticklabels=["No ASD Traits", "ASD Traits"],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)

    # Plot and save ROC curve if possible
    if auc is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close(fig)

    print("Saved confusion matrix plot and ROC curve (if available) to:")
    print(f" - {cm_path}")
    if auc is not None:
        print(f" - {roc_path}")

    # Extract the fitted preprocessor and model
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    # Build the feature name list as seen by the model after preprocessing
    numeric_cols = preprocessor.transformers_[0][2]
    ohe: OneHotEncoder = preprocessor.transformers_[1][1]
    cat_feature_names = ohe.get_feature_names_out(preprocessor.transformers_[1][2])

    all_feature_names = list(numeric_cols) + list(cat_feature_names)

    # Fit a separate scaler on the numeric+encoded features to match the app pattern
    # (The app expects a scaler + model working on an already one-hot-encoded frame.)
    scaler = StandardScaler()
    X_transformed = preprocessor.transform(X)
    scaler.fit(X_transformed)

    # Save artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(all_feature_names, FEATURE_NAMES_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved feature names to {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    train_and_save()
