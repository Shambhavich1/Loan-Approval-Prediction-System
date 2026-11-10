A small web app that predicts whether a loan application is likely to be approved.

Dataset contains columns as follows:
```json
{
  "no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
  "income_annum": 9600000, "loan_amount": 29900000, "loan_term": 12,
  "cibil_score": 778, "residential_assets_value": 2400000,
  "commercial_assets_value": 17600000, "luxury_assets_value": 22700000,
  "bank_asset_value": 8000000
}
```
- Target is encoded Approved = 1 (the notebook's `LabelEncoder` made Approved = 0), so the model's probability reads as "chance of approval".
- Random Forest uses 200 trees and a fixed seed for repeatable results.

## Notes

"What moved the result" swaps each input for a typical applicant's value and shows how
much the approval chance changes. It explains this model, not causation.
This is a decision aid, not a lending decision.
