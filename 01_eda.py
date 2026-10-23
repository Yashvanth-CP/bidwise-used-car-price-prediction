import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df1 = pd.read_csv("data/indian_used_cars_numeric_57058.csv")
df = pd.read_csv("data/indian_used_cars_raw_57480.csv")

#print(df1.shape)
print(df.shape)

#print(df1.head())
print(df.head())

#print(f"inforamtion : {df1.info()}")
print(df.info())

#print(f"Describing the data : {df1.describe()}")
print(df.describe())

#print(df1.isna().sum())
print(df.isna().sum())

df = df.drop(columns=["Unnamed: 0"])

print("1. looking at what km_driven really contains: \n")  

print(df["km_driven"].head())
print(df["km_driven"].sample(10, random_state=1))

print("2. duplicates : \n")
print(df.duplicated().sum())

print("3. missing fuel types \n ")

print(df[df["fuel_type"].isna()])
print(df["fuel_type"].value_counts())

print("4. price shape: \n ")
print(df["price"].describe())


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df["price"], bins=50)
ax[0].set_title("price")
ax[1].hist(np.log1p(df["price"]), bins=50)
ax[1].set_title("log1p(price)")
plt.show()

df["km_driven"] = (df["km_driven"]
                   .str.replace(",","")
                   .str.replace("km", "")
                   .str.strip()
                   .astype(float)
                )

print(df["km_driven"].dtype)
print(df["km_driven"].describe())
print(df.duplicated().sum())

print(df["price"].quantile([0.5, 0.9, 0.99, 0.999]))
print(df["km_driven"].quantile([0.5, 0.9, 0.99, 0.999]))

# looking at the extreme rows 

print(df.nlargest(10, "price")[["brand", "model", "manufacturing_year", "km_driven", "price"]])
print(df.nlargest(10, "km_driven")[["brand", "model", "manufacturing_year", "km_driven", "price"]])

# a feature  

CURRENT_YEAR = 2026

df["car_age"] = CURRENT_YEAR - df["manufacturing_year"]
df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)

print(df["km_per_year"].describe())
print(df["km_per_year"].quantile([0.5, 0.9, 0.99, 0.999]))
print(df.nlargest(10, "km_per_year")[["brand", "model", "manufacturing_year", "km_per_year","price" ]])

print((df["km_per_year"] > 100_000).sum())


# 1. Count rows above different cutoffs 

for cutoff in [50_000, 75_000, 100_000]:
    print(cutoff, (df["km_per_year"] > cutoff).sum())

# 2. Look at brand names

print(df["brand"].value_counts().head(30))
print(sorted(df["brand"].unique()))