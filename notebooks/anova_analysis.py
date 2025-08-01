import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Exploratory Data Analysis
# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Preview the first 5 rows
print("First 5 rows of the dataset :")
print(df.head())

# Rename columns
df.columns = df.columns.str.lower().str.replace(" ","_")

# Check for null values
print("\nNull value count per column :")
print(df.isnull().sum())

# Fill categorical values with mode and
# numberical values using median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            modeValue = df[col].mode()[0]
            df[col].fillna(modeValue,inplace = True)
            print(f"Filled missing values in {col} with mode : {modeValue}")
        else:
            medianValue = df[col].median()
            df[col].fillna(medianValue,inplace = True)
            print(f"Filled missing values in {col} with mode : {medianValue}")

# Convert categorical columns
categorical_cols = ["gender", "race/ethnicity", "parental_level_of_education","lunch", "test_preparation_course"]

for col in categorical_cols:
    df[col] = df[col].astype("category")

# Summary
print("\nDataframe Info:")
print(df.info())

print("\nStatistical Summary (Numerical and Categorical) : ")
print(df.describe(include="all"))