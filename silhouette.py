import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import os
import json

def silhouettescore(embedding, labels):
    return silhouette_score(embedding, labels)

def save_silhouette_scores(silhouette_scores, output_dir='unsupervised_output'):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'silhouette_scores.json')
    with open(json_path, 'w') as f:
        json.dump(silhouette_scores, f, indent=4)
    print(f"Silhouette scores saved!")

def silhouette_plot(embedding, labels, n_clusters, save_path=None):
    sample_silhouette_values = silhouette_samples(embedding, labels)

    fig, ax = plt.subplots()
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()
        
        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=silhouette_score(embedding, labels), color="red", linestyle="--")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
