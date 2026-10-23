import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


df = pd.read_csv("data/cars_clean.csv")
X = df.drop(columns=["price"])
y = np.log1p(df["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["manufacturing_year", "km_driven", "number_of_owners", "car_age", "km_per_year"]
cat_cols = ["brand", "model", "fuel_type", "transmission_type", "city", "bodytype"]

def make_prep():
   columnTransformer = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                              sparse_output=False), cat_cols),
    ])

   return columnTransformer

models = {
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                           n_jobs=-1, random_state=42),
    "hist_gb": HistGradientBoostingRegressor(random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("prep", make_prep()), ("model", model)])
    scores = -cross_val_score(pipe, X_train, y_train, cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=1)
    print(f"{name}: MAE = {scores.mean():.4f} ± {scores.std():.4f}")