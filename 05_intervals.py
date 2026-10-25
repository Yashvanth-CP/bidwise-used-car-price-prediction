import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/cars_clean.csv")
X = df.drop(columns=["price"])
y = np.log1p(df["price"])

# SAME split as before (same random_state=42), so the test set stays locked
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["manufacturing_year", "km_driven", "number_of_owners", "car_age", "km_per_year"]
cat_cols = ["brand", "model", "fuel_type", "transmission_type", "city", "bodytype"]

def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), cat_cols),
    ])



# 60% fit, 20% calibration, 20% validation (to check coverage)
X_fit, X_rest, y_fit, y_rest = train_test_split(X_train, y_train, test_size=0.4, random_state=1)
X_cal, X_val, y_cal, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=2)

pipe = Pipeline([("prep", make_prep()),
                 ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                                 n_jobs=-1, random_state=42))])
pipe.fit(X_fit, y_fit)

# errors on the calibration set
cal_errors = np.abs(y_cal - pipe.predict(X_cal))
n = len(cal_errors)
q = np.quantile(cal_errors, np.ceil((n + 1) * 0.90) / n)   # 90% interval

# check coverage on the validation set
pred = pipe.predict(X_val)
inside = (y_val >= pred - q) & (y_val <= pred + q)
print("q (log scale):", round(q, 4))
print("Coverage:", inside.mean())
print("Typical interval width:", round(2 * q, 4))

print(len(X_fit), len(X_cal), len(X_val))

pipe = Pipeline([
    ("prep", make_prep()),
    ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)),
])

pipe.fit(X_fit, y_fit)

cal_pred = pipe.predict(X_cal)
val_pred = pipe.predict(X_val)
cal_err = np.abs(y_cal - cal_pred).to_numpy()
val_err = np.abs(y_val - val_pred).to_numpy()

# 5 groups based on predicted price (edges learned from the calibration set)
edges = np.quantile(cal_pred, [0.2, 0.4, 0.6, 0.8])
cal_bin = np.digitize(cal_pred, edges)    # 0,1,2,3,4
val_bin = np.digitize(val_pred, edges)

# one q per group, from the calibration errors of that group
q_bins = np.array([np.quantile(cal_err[cal_bin == b], 0.90) for b in range(5)])
print("q per group:", q_bins.round(3))

# check coverage on the validation set
inside = val_err <= q_bins[val_bin]
print("Overall coverage:", inside.mean())
for b in range(5):
    print(f"group {b}: coverage = {inside[val_bin == b].mean():.3f}, q = {q_bins[b]:.3f}")