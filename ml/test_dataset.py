import pandas as pd
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Dataset location
dataset_path = project_folder / "data" / "IoTID20.csv"

print("Dataset path:")
print(dataset_path)

# Check whether the file exists
if not dataset_path.exists():
    print("\nERROR: IoTID20.csv was not found!")
    print("Please check that the file is inside the data folder.")
    exit()

# Load dataset
df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())