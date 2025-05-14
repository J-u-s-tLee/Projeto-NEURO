import umap
import numpy as np 
import pandas as pd
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def dimensionality_reduce(data, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )

    embedding = reducer.fit_transform(data)
    return embedding

""" def dbscan_clustering(embedding, eps=0.1, min_samples=30):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embedding)
    return labels """

def dbscan_find_3_clusters(embedding, 
                           eps_range=np.linspace(0.05, 1.0, 5), 
                           min_samples_range=[30, 40, 50, 60, 70]):
    for min_samples in min_samples_range:
        for eps in eps_range:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embedding)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters == 3:
                print(f"Found 3 clusters with eps={eps:.2f}, min_samples={min_samples}")
                return labels
    print("Could not find combination that results in 3 clusters.")
    return DBSCAN(eps=0.5, min_samples=10).fit_predict(embedding) 

def remove_outliers(data, labels):

    non_outliers = labels != -1
    filtered_data = data[non_outliers] 
    filtered_labels = labels[non_outliers]
    return filtered_data, filtered_labels

def plot_embedding(embedding, labels=None, figsize=(8, 6), point_size=10, cmap='plasma'):

    plt.figure(figsize=figsize)

    if labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=point_size)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], color='gray', s=point_size)
    
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()
