"""Flask backend for the Loan Approval Predictor.

Routes
    GET  /              the web app
    GET  /api/health    liveness check
    GET  /api/meta      model metrics, feature importance, dataset summary
    POST /api/predict   score one application
"""
from __future__ import annotations

import logging
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loan-app")

app = Flask(__name__, static_folder="static", static_url_path="/static")


def load_bundle() -> dict:
    if not train.MODEL_PATH.exists():
        log.info("No trained model found, training one now.")
        train.main(train.DEFAULT_CSV)
    return joblib.load(train.MODEL_PATH)


BUNDLE = load_bundle()
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["features"]
BASELINE = BUNDLE["baseline"]

LABELS = {
    "no_of_dependents": "Dependents",
    "education": "Education",
    "self_employed": "Self-employment",
    "income_annum": "Annual income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "cibil_score": "CIBIL score",
    "residential_assets_value": "Residential assets",
    "commercial_assets_value": "Commercial assets",
    "luxury_assets_value": "Luxury assets",
    "bank_asset_value": "Bank assets",
}

# field -> (min, max, whole number only)
NUMERIC_RULES = {
    "no_of_dependents": (0, 10, True),
    "income_annum": (1, 1e10, False),
    "loan_amount": (1, 1e10, False),
    "loan_term": (1, 30, True),
    "cibil_score": (300, 900, True),
    "residential_assets_value": (0, 1e11, False),
    "commercial_assets_value": (0, 1e11, False),
    "luxury_assets_value": (0, 1e11, False),
    "bank_asset_value": (0, 1e11, False),
}


def validate(payload: dict) -> tuple[dict, dict]:
    """Return (clean_values, errors)."""
    clean, errors = {}, {}

    for field, (lo, hi, whole) in NUMERIC_RULES.items():
        raw = payload.get(field)
        if raw in (None, ""):
            errors[field] = "Enter a value."
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[field] = "Enter a number."
            continue
        if whole and value != int(value):
            errors[field] = "Use a whole number."
        elif not lo <= value <= hi:
            errors[field] = f"Must be between {lo:,.0f} and {hi:,.0f}."
        else:
            clean[field] = value

    edu = payload.get("education")
    if edu not in train.EDUCATION_MAP:
        errors["education"] = "Choose Graduate or Not Graduate."
    else:
        clean["education"] = train.EDUCATION_MAP[edu]

    emp = payload.get("self_employed")
    if emp not in train.SELF_EMPLOYED_MAP:
        errors["self_employed"] = "Choose Yes or No."
    else:
        clean["self_employed"] = train.SELF_EMPLOYED_MAP[emp]

    return clean, errors


def approval_probability(row: dict) -> float:
    frame = pd.DataFrame([row], columns=FEATURES)
    return float(MODEL.predict_proba(frame)[0][1])


def explain(row: dict, base_p: float) -> list[dict]:
    """Per-feature effect: how much the approval chance drops when that feature
    is swapped for a typical (median) applicant value."""
    out = []
    for feature in FEATURES:
        swapped = dict(row)
        swapped[feature] = BASELINE[feature]
        impact = base_p - approval_probability(swapped)
        out.append(
            {
                "feature": feature,
                "label": LABELS[feature],
                "impact": round(impact * 100, 1),
            }
        )
    out.sort(key=lambda f: abs(f["impact"]), reverse=True)
    return [f for f in out if abs(f["impact"]) >= 0.1][:6]


def risk_band(p: float) -> dict:
    if p >= 0.80:
        return {"key": "low", "label": "Low risk"}
    if p >= 0.50:
        return {"key": "moderate", "label": "Moderate risk"}
    if p >= 0.30:
        return {"key": "elevated", "label": "Elevated risk"}
    return {"key": "high", "label": "High risk"}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=BUNDLE["meta"]["selected_model"])


@app.get("/api/meta")
def meta():
    return jsonify(BUNDLE["meta"])


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the application as a JSON object."), 400

    clean, errors = validate(payload)
    if errors:
        return jsonify(error="Some fields need attention.", fields=errors), 422

    p = approval_probability(clean)
    total_assets = sum(
        clean[k]
        for k in (
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        )
    )
    approved = p >= 0.5
    log.info("prediction approved=%s p=%.3f", approved, p)

    return jsonify(
        approved=approved,
        probability=round(p, 4),
        verdict="Likely to be approved" if approved else "Likely to be rejected",
        risk=risk_band(p),
        factors=explain(clean, p),
        ratios={
            "loan_to_income": round(clean["loan_amount"] / clean["income_annum"], 2),
            "asset_cover": round(total_assets / clean["loan_amount"], 2),
            "total_assets": total_assets,
        },
        model=BUNDLE["meta"]["selected_model"],
    )


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found."), 404


@app.errorhandler(500)
def server_error(exc):
    log.exception("Unhandled error: %s", exc)
    return jsonify(error="Something went wrong on the server."), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
