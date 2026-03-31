import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from pipeline import extract, load, transform, run_pipeline

df = run_pipeline()
print(df.head())

X = df.iloc[:,[1,2]].values
print(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,6))
plt.style.use('ggplot')
plt.plot(range(1,11), wcss, marker = 'o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCCS")
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K = {i}, Silhouette Score = {score:.4f}")
plt.figure(figsize=(10,6))
plt.style.use('ggplot')
plt.plot(range(2,11), silhouette_scores, marker='o')
plt.title("Silhouette Score Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()  

# From the above elbow and silhouette score its clear that the number oof cluster is 4
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state=42) 
print(kmeans)        

cluster = kmeans.fit_predict(X_scaled)
print(cluster)

plt.scatter(X_scaled[cluster == 0,0], X_scaled[cluster == 0,1], s = 100, label = 'Cluster 0')
plt.scatter(X_scaled[cluster == 1,0], X_scaled[cluster == 1,1], s = 100, label = 'Cluster 1')
plt.scatter(X_scaled[cluster == 2,0], X_scaled[cluster == 2,1], s = 100, label = 'Cluster 2')
plt.scatter(X_scaled[cluster == 3,0], X_scaled[cluster == 3,1], s = 100, label = 'Cluster 3')
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:,0], centroids[:,1], s= 100, label = 'Centroid', marker = 'X')
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

print("Visualising without scaled")
plt.scatter(X[cluster == 0,0], X[cluster == 0,1], s = 100, label = 'Cluster 0')
plt.scatter(X[cluster == 1,0], X[cluster == 1,1], s = 100, label = 'Cluster 1')
plt.scatter(X[cluster == 2,0], X[cluster == 2,1], s = 100, label = 'Cluster 2')
plt.scatter(X[cluster == 3,0], X[cluster == 3,1], s = 100, label = 'Cluster 3')
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(centroids[:,0], centroids[:,1], s= 100, label = 'Centroid', marker = 'X')
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

df['Cluster'] = cluster

print(df.head())