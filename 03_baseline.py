import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data/cars_clean.csv")
X = df.drop(columns=["price"])
y = np.log1p(df["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ["manufacturing_year", "km_driven", "number_of_owners", "car_age", "km_per_year"]
cat_cols = ["brand", "model", "fuel_type", "transmission_type", "city", "bodytype"]

preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
                          ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat_cols),

    ]
)


baseline = Pipeline([("prep", preprocess), ("model", DummyRegressor(strategy="median"))])
ridge = Pipeline([("prep", preprocess), ("model", Ridge(alpha=1.0))])

for name, m in [("dummy", baseline), ("ridge", ridge)]:
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(name, "MAE (log scale):", mean_absolute_error(y_test, pred))