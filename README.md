# Telco Customer Churn Prediction
**Problem:** Who is about to leave, and what should we do about it?
**Data:** IBM Telco Customer Churn (Kaggle, ~7k customers). Download `Telco-Customer-Churn.csv` into `data/`.
**Tools:** Python, pandas, scikit-learn.

## Approach
1. Clean `TotalCharges` (blanks = new customers).
2. Compare logistic regression vs gradient boosting, evaluated by ROC-AUC.
3. Lower decision threshold to 0.4 to catch more churners.
4. Estimate annual value at risk to prioritize a retention campaign.

## Key findings (fill with your results)
- Month-to-month contracts churn at X% vs Y% for 2-year.
- Fiber optic and no-tech-support customers churn more.
- Best model AUC: X.

## Recommendation
Target the top-risk decile with a contract-upgrade offer.

## Run
```
pip install -r requirements.txt
python churn.py
```
