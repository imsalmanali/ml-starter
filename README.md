# ml-starter

A small Python machine-learning starter: train a scikit-learn classifier on the Iris dataset, save the model, and run predictions.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python src/train.py
```

This writes `models/iris_classifier.joblib` and prints accuracy on a held-out test set.

## Predict

```powershell
python src/predict.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

## Notebook

Open `notebooks/quickstart.ipynb` after activating the virtual environment:

```powershell
jupyter notebook notebooks/quickstart.ipynb
```
