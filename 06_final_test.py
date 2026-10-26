import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# ---------- 1. Data and splits (identical to 05_intervals.py) ----------
df = pd.read_csv("data/cars_clean.csv")
X = df.drop(columns=["price"])
y = np.log1p(df["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_fit, X_rest, y_fit, y_rest = train_test_split(X_train, y_train, test_size=0.4, random_state=1)
X_cal, X_val, y_cal, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=2)

# ---------- 2. Pipeline builder (defined BEFORE it is used) ----------
num_cols = ["manufacturing_year", "km_driven", "number_of_owners", "car_age", "km_per_year"]

def build_pipe(drop=(), **rf_params):
    nums = [c for c in num_cols if c not in drop]
    cats = [c for c in ["model", "fuel_type", "transmission_type", "city", "bodytype"]
            if c not in drop]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), nums),
        ("brand", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["brand"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), cats),
    ])
    params = dict(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)
    params.update(rf_params)
    return Pipeline([("prep", prep), ("model", RandomForestRegressor(**params))])

# ---------- 3. Train and calibrate (no test data used) ----------
final = build_pipe(drop=("city",))
final.fit(X_fit, y_fit)

cal_pred = final.predict(X_cal)
cal_err = np.abs(y_cal - cal_pred).to_numpy()
edges = np.quantile(cal_pred, [0.2, 0.4, 0.6, 0.8])
cal_bin = np.digitize(cal_pred, edges)
q_bins = np.array([np.quantile(cal_err[cal_bin == b], 0.90) for b in range(5)])
print("q per group:", q_bins.round(3))

# ---------- 4. The ONE look at the test set ----------
test_pred = final.predict(X_test)
test_err = np.abs(y_test - test_pred).to_numpy()
test_bin = np.digitize(test_pred, edges)

print("Test MAE (log):", round(test_err.mean(), 4))
print("Test MAE in % (approx):", round((np.exp(test_err.mean()) - 1) * 100, 1))
print("Median abs error in rupees:",
      round(np.median(np.abs(np.expm1(y_test) - np.expm1(test_pred)))))
print("Interval coverage on test:", round((test_err <= q_bins[test_bin]).mean(), 3))
for b in range(5):
    print(f"group {b}: coverage = {(test_err[test_bin == b] <= q_bins[b]).mean():.3f}")

    # ---------- 5. Charts ----------
import os
import matplotlib.pyplot as plt
os.makedirs("reports/figures", exist_ok=True)

# Chart 1: predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_pred, s=3, alpha=0.3)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual log1p(price)")
plt.ylabel("Predicted log1p(price)")
plt.title("Predicted vs actual (test set)")
plt.savefig("reports/figures/pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("saved chart 1")

# Chart 2: error and coverage by predicted-price group
names = ["cheapest", "low", "mid", "high", "luxury"]
mae_g = [test_err[test_bin == b].mean() for b in range(5)]
cov_g = [(test_err[test_bin == b] <= q_bins[b]).mean() for b in range(5)]

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(names, mae_g, color="steelblue")
ax1.set_ylabel("MAE (log scale)")
ax2 = ax1.twinx()
ax2.plot(names, cov_g, "ro-")
ax2.axhline(0.90, color="gray", linestyle="--")
ax2.set_ylim(0.8, 1.0)
ax2.set_ylabel("Coverage (dashed line = 90% target)")
plt.title("Error and interval coverage by predicted-price group")
plt.savefig("reports/figures/error_by_group.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Chart 3: what the intervals look like for 40 random test cars
rng = np.random.default_rng(0)
idx = rng.choice(len(y_test), 40, replace=False)
idx = idx[np.argsort(test_pred[idx])]
actual = y_test.to_numpy()[idx]
pred = test_pred[idx]
q_i = q_bins[test_bin[idx]]
x = np.arange(40)
inside = (actual >= pred - q_i) & (actual <= pred + q_i)

plt.figure(figsize=(10, 5))
plt.vlines(x, np.expm1(pred - q_i), np.expm1(pred + q_i),
           color="steelblue", alpha=0.6, label="90% interval")
plt.scatter(x, np.expm1(pred), color="steelblue", s=15, label="Prediction")
plt.scatter(x[inside], np.expm1(actual[inside]), color="green", s=22, label="Actual price (inside)")
plt.scatter(x[~inside], np.expm1(actual[~inside]), color="red", s=22, label="Actual price (outside)")
plt.yscale("log")
plt.xlabel("40 random test cars, sorted by predicted price")
plt.ylabel("Price in rupees (log axis)")
plt.legend()
plt.title("90% prediction intervals on unseen cars")
plt.savefig("reports/figures/interval_examples.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("saved charts 2 and 3")