import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/indian_used_cars_raw_57480.csv")
df = df.drop(columns=["Unnamed: 0"])

# 1. km_driven text -> number 

df["km_driven"] = df["km_driven"].str.replace(",", "").astype(float)


# Fix brand spelling 
brand_map = {
    "Maruti": "Maruti Suzuki",
    "Mahindra-Renault": "Mahindra Renault",
}
df["brand"] = df["brand"].replace(brand_map)

# 3. removing impossible km values 
df["car_age"] = 2026 - df["manufacturing_year"]
df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)
df = df[df["km_per_year"] <= 100_000].copy()

# 4. Drop the 28 rows with missing fuel_type 
df = df.dropna(subset=["fuel_type"])
print(df.shape)

# exploring model column 

print(df["model"].unique())
print(df["model"].sample(15, random_state=1).tolist())
print(df["model"].nunique())
print(df["brand"].nunique())


# 1. remove [year] brackets
df["model"] = df["model"].str.replace(r"\[.*?\]","", regex=True)

# 2. lowercase and tidy spaces
df["model"] = df["model"].str.lower().str.replace(r"\s+"," ",regex=True).str.strip()

# 3. remove the brand name from the start of the model

def strip_brand(model, brand):
    brand = brand.lower()
    first_word = brand.split()[0]
    if model.startswith(brand):
        return model[len(brand):].strip()
    if model.startswith(first_word):
        return model[len(first_word):].strip()
    return model

df["model"] = [strip_brand(m, b) for m, b in zip(df["model"], df["brand"])]

print(" unique : \n")
print(df["model"].nunique())
print(df["model"].value_counts().head(20))
print((df["model"].value_counts() < 5).sum()) 

# splitting the data 

# saving cleaned data 

df.to_csv("data/cars_clean.csv", index=False)

X = df.drop(columns=["price", "km_per_year"])
y = np.log1p(df["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.describe())
print(y_test.describe())
