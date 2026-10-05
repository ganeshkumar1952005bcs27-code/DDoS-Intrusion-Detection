import pandas as pd
from pathlib import Path

# Find project folder
project_folder = Path(__file__).resolve().parent.parent

# Dataset path
dataset_path = project_folder / "data" / "IoTID20.csv"

# Load dataset
df = pd.read_csv(dataset_path)

print("=" * 60)
print("IoTID20 DATASET INSPECTION")
print("=" * 60)

# 1. Dataset size
print("\n1. DATASET SIZE")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# 2. Label distribution
print("\n2. LABEL DISTRIBUTION")
print(df["Label"].value_counts())

# 3. Category distribution
print("\n3. ATTACK CATEGORY DISTRIBUTION")
print(df["Cat"].value_counts())

# 4. Sub-category distribution
print("\n4. ATTACK SUB-CATEGORY DISTRIBUTION")
print(df["Sub_Cat"].value_counts())

# 5. Missing values
print("\n5. MISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

# 6. Infinite values
print("\n6. INFINITE VALUES")

numeric_df = df.select_dtypes(include="number")

infinite_count = numeric_df.isin([float("inf"), float("-inf")]).sum().sum()

print("Total infinite values:", infinite_count)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)