import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Dict, Any

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "resume_scorer.joblib")

def train(train_csv_path: str, model_type: str = "rf"):
    """
    Expects a CSV with columns: years_exp, skill_count, format_score, num_experience_items, label
    label: integer or binary (e.g., score bucket or hireable: 0/1)
    """
    df = pd.read_csv(train_csv_path)
    features = ["years_exp", "skill_count", "format_score", "num_experience_items"]
    if not all(c in df.columns for c in features + ["label"]):
        raise ValueError(f"Train csv must contain columns: {features + ['label']}")
    X = df[features].fillna(0)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("=== Classification report on held-out set ===")
    print(classification_report(y_test, preds))
    joblib.dump((model, features), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first with scorer.train(train_csv_path).")
    model, features = joblib.load(MODEL_PATH)
    return model, features


def predict(features: Dict[str, Any]):
    model, feature_list = load_model()
    X = [features.get(f, 0) for f in feature_list]
    score_label = model.predict([X])[0]
    # also provide probabilities if possible
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([X])[0].max()
    return {"label": int(score_label), "confidence": float(prob) if prob is not None else None}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, help="Path to train CSV")
    args = parser.parse_args()
    if args.train_csv:
        train(args.train_csv)
    else:
        print("Provide --train_csv to train a scorer model.")