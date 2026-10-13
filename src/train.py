"""Train a simple Iris classifier and save it to models/."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_classifier.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )


def main() -> None:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target_names": list(iris.target_names)}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
