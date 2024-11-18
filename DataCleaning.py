# Apply data cleaning techniques on any dataset (e,g, wine dataset). Techniques may include handling missing values, outliers, inconsistent values. A set of validation rules can be prepared based on the dataset and validations can be performed.

import pandas as pd
import numpy as np

# Step 1: Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, delimiter=';')

# Step 2: Display the first few rows of the dataset
print("Initial Dataset Snapshot:\n", df.head())

# Step 3: Check for missing values
print("\nMissing Values Count:\n", df.isnull().sum())

# Step 4: Handling missing values (if any)
df.fillna(df.mean(), inplace=True)

# Step 5: Detecting outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nData shape after removing outliers:", df_no_outliers.shape)

# Step 6: Handling inconsistent values (e.g., checking pH values range)
df_no_outliers = df_no_outliers[(df_no_outliers['pH'] >= 0) & (df_no_outliers['pH'] <= 14)]

# Step 7: Display cleaned dataset summary
print("\nCleaned Dataset Summary:\n", df_no_outliers.describe())

# Step 8: Save cleaned data to a new CSV file
df_no_outliers.to_csv('cleaned_wine_quality.csv', index=False)
print("\nCleaned data saved as 'cleaned_wine_quality.csv'")
