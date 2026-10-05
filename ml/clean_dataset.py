import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Locate project and dataset
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
dataset_path = project_folder / "data" / "IoTID20.csv"

print("=" * 60)
print("IoTID20 DATASET CLEANING")
print("=" * 60)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(dataset_path)

print("\nOriginal dataset:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# --------------------------------------------------
# 3. Replace infinite values with NaN
# --------------------------------------------------

print("\nReplacing infinite values...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Count missing values after replacing infinity
missing_before = df.isnull().sum().sum()

print("Missing values after infinity replacement:",
      missing_before)

# --------------------------------------------------
# 4. Fill missing numeric values with median
# --------------------------------------------------

print("\nFilling missing numeric values with median...")

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].median())

# --------------------------------------------------
# 5. Check the result
# --------------------------------------------------

missing_after = df.isnull().sum().sum()

print("\nMissing values after cleaning:",
      missing_after)

# --------------------------------------------------
# 6. Check infinite values
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

infinite_after = np.isinf(numeric_df.to_numpy()).sum()

print("Infinite values after cleaning:",
      infinite_after)

# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

output_path = project_folder / "data" / "IoTID20_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:")
print(output_path)

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)