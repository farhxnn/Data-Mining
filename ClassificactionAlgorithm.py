# 4. Use Naive bayes, K-nearest, and Decision tree classification algorithms and build classifiers on 
# any two datasets. Divide the data set into training and test set. Compare the accuracy of the 
# different classifiers under the following situations: 
# I. a) Training set = 75% Test set = 25% b) Training set = 66.6% (2/3rd of total), Test set = 33.3% 
# II. Training set is chosen by i) hold out method ii) Random subsampling iii) Cross-Validation. 
# Compare the accuracy of the classifiers obtained. Data needs to be scaled to standard format

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load datasets
from sklearn.datasets import load_iris, load_wine

# Prepare datasets
iris = load_iris()
wine = load_wine()

datasets = [
    ('Iris', pd.DataFrame(iris.data, columns=iris.feature_names), pd.Series(iris.target)),
    ('Wine', pd.DataFrame(wine.data, columns=wine.feature_names), pd.Series(wine.target))
]

# Classifier models
models = {
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier()
}

# Function to train and evaluate models
def evaluate_models(X, y, test_size, method):
    print(f"\n--- Evaluating Models with {method} (Test size: {test_size * 100}%) ---")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hold-out method
    if method == 'Hold-out':
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
    # Random subsampling
    elif method == 'Random Subsampling':
        rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_index, test_index in rs.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Cross-validation
    elif method == 'Cross-Validation':
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5)
            print(f"{name} - Cross-Validation Accuracy: {np.mean(scores):.2f}")
        return

    # Train and evaluate models for hold-out and subsampling
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Accuracy: {accuracy:.2f}")

# Run evaluation on each dataset with different configurations
for dataset_name, X, y in datasets:
    print(f"\n\n### Dataset: {dataset_name} ###")

    # I. a) Training set = 75%, Test set = 25%
    evaluate_models(X, y, test_size=0.25, method='Hold-out')
    evaluate_models(X, y, test_size=0.25, method='Random Subsampling')
    evaluate_models(X, y, test_size=0.25, method='Cross-Validation')

    # I. b) Training set = 66.6%, Test set = 33.3%
    evaluate_models(X, y, test_size=0.333, method='Hold-out')
    evaluate_models(X, y, test_size=0.333, method='Random Subsampling')
    evaluate_models(X, y, test_size=0.333, method='Cross-Validation')
