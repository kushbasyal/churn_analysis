from paths import raw_data_path,clean_data_path, clustering_path
from loader import load_data
from cleaner import inspect_data, clean_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

def find_cluster(data):
    # Scale the input features
    data_scaled = StandardScaler().fit_transform(data)
    # Using Elbow Method to identify the number of clusters
    wcss =[]
    for i in range(2,11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state= 42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    plt.style.use('ggplot')
    plt.figure(figsize =(7,4))
    plt.plot(range(2,11), wcss, marker ='o')
    plt.title("The Elbow Method")
    plt.xlabel("No of Clusters")
    plt.ylabel("WCSS")
    elbow_plot = os.path.join(clustering_path, "elbow_method.png")
    plt.savefig(elbow_plot, dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

    # Use Silhouette Score to identyfying the number of cluster
    sil_score =[]
    for k in range (2,11):
        kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        sil_score.append(score)
        print(f"Clusters: {k} Score: {score}")
    plt.style.use('ggplot')
    plt.figure(figsize =(7,4))
    plt.plot(range(2,11), sil_score, marker ='o')
    plt.title("The Silhouette Method")
    plt.xlabel("No of Clusters")
    plt.ylabel("Sil_Score")
    silhouette_plot = os.path.join(clustering_path, "silhouette_method.png")
    plt.savefig(silhouette_plot, dpi =300, bbox_inches ='tight')
    plt.show()

def visualise_cluster(data, k):
    # Scaled the data and fit kmeans
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters = k, init='k-means++', random_state = 42)
    clusters = kmeans.fit_predict(data_scaled)

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8,6))
    sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue = clusters, palette= "tab10", s= 100)

    # Transform centroids back to raw data and plot them on top
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    plt.scatter(centroids[:,0], centroids[:,1], color = 'black', marker = 'X', label = 'Centroids')
    plt.title(f"Clusters (k = {k})")
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.tight_layout()
    plt.savefig(os.path.join(clustering_path, f"cluster_plot_k{k}.png"),dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()

   

