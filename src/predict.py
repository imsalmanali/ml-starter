"""Load the saved Iris classifier and predict a species from four measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Iris species from flower measurements.")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run python src/train.py first.")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    target_names = bundle["target_names"]

    features = np.array(
        [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]]
    )
    pred = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    print(f"Predicted species: {target_names[pred]}")
    for name, p in zip(target_names, proba, strict=True):
        print(f"  {name}: {p:.3f}")


if __name__ == "__main__":
    main()
