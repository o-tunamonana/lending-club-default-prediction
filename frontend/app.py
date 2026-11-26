"""Web form for loan officers. It only talks to the model API, never to the model itself."""

import os

import requests
from flask import Flask, render_template, request

API_URL = os.getenv("API_URL", "http://localhost:8000")

EMP_LENGTH = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
              "6 years", "7 years", "8 years", "9 years", "10+ years", "NI"]
PURPOSE = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
           "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
           "renewable_energy", "educational"]
HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
STATES = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
          "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
          "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
          "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]

DEFAULTS = {"revenue": "65000", "dti_n": "18.5", "loan_amnt": "12000", "fico_n": "700",
            "emp_length": "5 years", "purpose": "debt_consolidation",
            "home_ownership_n": "RENT", "addr_state": "NY"}

LABELS = {"revenue": "Annual income", "dti_n": "Debt-to-income ratio", "loan_amnt": "Loan amount",
          "fico_n": "FICO score", "emp_length": "Employment length", "purpose": "Loan purpose",
          "home_ownership_n": "Home ownership", "addr_state": "State"}

app = Flask(__name__)


def build_payload(form) -> tuple[dict, list[str]]:
    """Convert form strings to numbers; collect friendly errors instead of crashing."""
    payload, errors = {}, []
    for field in ["revenue", "dti_n", "loan_amnt", "fico_n"]:
        raw = form.get(field, "").replace(",", "").strip()
        try:
            payload[field] = float(raw)
        except ValueError:
            errors.append(f"{LABELS[field]} must be a number.")
    for field in ["emp_length", "purpose", "home_ownership_n", "addr_state"]:
        payload[field] = form.get(field, "")
    return payload, errors


def explain_api_errors(body: dict) -> list[str]:
    messages = []
    for item in body.get("detail", []) if isinstance(body.get("detail"), list) else []:
        field = item.get("loc", ["", ""])[-1]
        messages.append(f"{LABELS.get(field, field)}: {item.get('msg', 'invalid value')}.")
    return messages or [str(body.get("detail", "The request was rejected."))]


@app.route("/", methods=["GET", "POST"])
def index():
    values, result, errors = dict(DEFAULTS), None, []
    if request.method == "POST":
        values = {k: request.form.get(k, "") for k in DEFAULTS}
        payload, errors = build_payload(request.form)
        if not errors:
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                elif response.status_code == 422:
                    errors = explain_api_errors(response.json())
                elif response.status_code == 503:
                    errors = ["The model is still loading. Please try again in a minute."]
                else:
                    errors = [f"The model service returned an error ({response.status_code})."]
            except requests.RequestException:
                errors = ["The model service is unreachable right now. Please try again later."]
    return render_template("index.html", values=values, result=result, errors=errors,
                           emp_length=EMP_LENGTH, purpose=PURPOSE, home=HOME, states=STATES)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
