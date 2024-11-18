# Let's implement K-means clustering on a dataset using Python. In this task, we'll:

# Use the Iris dataset for simplicity.
# Apply the K-means algorithm with varying parameters (e.g., number of clusters).
# Measure performance using Mean Squared Error (MSE).
# Plot the MSE over iterations for a set of parameters to visualize convergence.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Load the Iris dataset
data = load_iris()
X = data.data

# Function to calculate MSE for a given set of labels and cluster centers
def calculate_mse(X, labels, centers):
    mse = 0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        mse += np.sum((cluster_points - center) ** 2)
    return mse / len(X)

# Function to perform K-means clustering and plot MSE over iterations
def kmeans_clustering(X, n_clusters, max_iter=100, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, n_init=10)
    kmeans.fit(X)
    
    # Store MSE after each iteration
    mse_list = []
    for i in range(1, len(kmeans.n_iter_) + 1):
        labels = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        mse = calculate_mse(X, labels, centers)
        mse_list.append(mse)
    
    # Plot MSE over iterations
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mse_list) + 1), mse_list, marker='o', linestyle='-', color='blue')
    plt.title(f'MSE Over Iterations (K = {n_clusters})')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()
    
    return kmeans

# Run K-means with different parameters and plot MSE
n_clusters_list = [2, 3, 4, 5]  # Varying number of clusters
for n_clusters in n_clusters_list:
    print(f"\n### Running K-means with K = {n_clusters} ###")
    kmeans_model = kmeans_clustering(X, n_clusters)

    # Display final cluster centers and inertia
    print(f"Cluster Centers:\n{kmeans_model.cluster_centers_}")
    print(f"Final MSE (Inertia): {kmeans_model.inertia_:.2f}")
    print(f"Number of iterations: {kmeans_model.n_iter_}")
