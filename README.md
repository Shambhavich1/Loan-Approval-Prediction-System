# Lendlens: Loan Approval Predictor

A small web app that predicts whether a loan application is likely to be approved.
It turns `Loan_Approval_Prediction.ipynb` into a trained model, a Flask API and a browser UI.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Put your dataset at data/loan_approval_dataset.csv
# 2. Train the model (same steps as the notebook)
python train.py

# 3. Start the app
python app.py          # http://127.0.0.1:5000
```

If `data/loan_approval_dataset.csv` is missing, the app trains on generated sample
data so you can try the interface. A banner in the UI tells you when this is the case.
Production: `gunicorn app:app --bind 0.0.0.0:8000`.

## Project layout

```
app.py            Flask API + serves the UI
train.py          Notebook pipeline: clean, encode, split, train, compare, save
model/            loan_model.joblib (created by train.py)
data/             put loan_approval_dataset.csv here
static/           index.html, styles.css, app.js
```

## API

| Route | Method | What it does |
|---|---|---|
| `/api/predict` | POST | Scores one application. Returns probability, verdict, risk band, key factors, ratios |
| `/api/meta` | GET | Model metrics, confusion matrix, feature importance, dataset summary |
| `/api/health` | GET | Liveness check |

Example request body:

```json
{
  "no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
  "income_annum": 9600000, "loan_amount": 29900000, "loan_term": 12,
  "cibil_score": 778, "residential_assets_value": 2400000,
  "commercial_assets_value": 17600000, "luxury_assets_value": 22700000,
  "bank_asset_value": 8000000
}
```

Invalid input returns HTTP 422 with a `fields` object of per-field messages.

## Differences from the notebook

- Target is encoded Approved = 1 (the notebook's `LabelEncoder` made Approved = 0), so the model's probability reads as "chance of approval".
- Random Forest uses 200 trees and a fixed seed for repeatable results.
- Logistic Regression gets a `StandardScaler` so it converges on rupee-sized values.
- Text values are trimmed (the dataset has leading spaces) and the encoders are fixed maps, so the app encodes input exactly as training did.

## Notes

"What moved the result" swaps each input for a typical applicant's value and shows how
much the approval chance changes. It explains this model, not causation.
This is a decision aid, not a lending decision.
