import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID", "Churn"])
print(df.groupby("Contract")["churn"].mean().round(3))

X, y = df.drop(columns="churn"), df["churn"]
num = X.select_dtypes("number").columns
cat = X.select_dtypes("object").columns
pre = ColumnTransformer([("n", StandardScaler(), num),
                         ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {"logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
          "gbm": GradientBoostingClassifier(random_state=42)}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    print(name, "AUC:", round(roc_auc_score(y_te, proba), 3))
    print(classification_report(y_te, proba > 0.4))

lr = Pipeline([("pre", pre), ("model", models["logreg"])]).fit(X_tr, y_tr)
names = lr.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
print(coef.head(5), coef.tail(5))

# Retention campaign value (assumes $20 offer cost)
te = X_te.copy(); te["p"] = proba
te["value_at_risk"] = te["MonthlyCharges"] * 12 * te["p"]
print("Value at risk, top 100:", te.sort_values("value_at_risk", ascending=False).head(100)["value_at_risk"].sum())
