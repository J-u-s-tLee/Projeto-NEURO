from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def silhouette_method(data, cluster_range):
    silhouette_avgs = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_avgs.append(silhouette_avg)

    plt.figure(figsize=(6, 4))
    plt.plot(cluster_range, silhouette_avgs, marker='o')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Silhouette Mean')
    plt.title('Silhouette Method')
    
    optimal_k = cluster_range[np.argmax(silhouette_avgs)]
    plt.axvline(x=optimal_k, linestyle='--', color='blue', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.show()

    return optimal_k


def elbow_method(data, range):
    
    distortion = []
    for k in range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortion.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range, distortion, marker='o')
    plt.xlabel('Number of Cluster k')
    plt.ylabel('Distortion (SSE)')
    plt.title('Elbow Method')
    plt.xticks(range)
    plt.show()

def Kmeans_method(data, k):

    k_means = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    cluster_labels = k_means.fit_predict(data)

    return cluster_labels
