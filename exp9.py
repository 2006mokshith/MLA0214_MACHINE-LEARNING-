# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Sample dataset
X = np.array([[1],[2],[3],[8],[9],[10]])

# Create EM model with 2 clusters
em = GaussianMixture(n_components=2, random_state=0)

# Fit the model
em.fit(X)

# Predict cluster labels
labels = em.predict(X)

# Print results
print("Data Points:", X.flatten())
print("Cluster Labels:", labels)

# Plot results
plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis', s=100)
plt.title("Expectation Maximization Clustering")
plt.xlabel("Data Points")
plt.show()