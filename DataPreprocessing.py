# Below is a simple Python program that demonstrates data preprocessing techniques like standardization, normalization, transformation, aggregation, discretization, and sampling using the Iris dataset.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Step 1: Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=columns)

# Step 2: Display the initial dataset snapshot
print("Initial Dataset:\n", df.head())

# Step 3: Standardization (Z-score scaling)
scaler = StandardScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
print("\nAfter Standardization:\n", df.head())

# Step 4: Normalization (Min-Max scaling)
scaler = MinMaxScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
print("\nAfter Normalization:\n", df.head())

# Step 5: Transformation (Log transformation for petal_length)
df['log_petal_length'] = np.log1p(df['petal_length'])
print("\nAfter Log Transformation (petal_length):\n", df[['petal_length', 'log_petal_length']].head())

# Step 6: Aggregation (Average sepal dimensions by species)
aggregated_data = df.groupby('species').agg({'sepal_length': 'mean', 'sepal_width': 'mean'})
print("\nAggregated Data (Average Sepal Dimensions by Species):\n", aggregated_data)

# Step 7: Discretization (Binarization of petal_width)
discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
df['petal_width_binned'] = discretizer.fit_transform(df[['petal_width']])
print("\nAfter Binarization (petal_width):\n", df[['petal_width', 'petal_width_binned']].head())

# Step 8: Sampling (Splitting the dataset into training and testing sets)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print("\nTraining Set Size:", train_df.shape)
print("Testing Set Size:", test_df.shape)
