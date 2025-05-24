import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_find_3_clusters(original_data, embedding, 
                           eps_range=np.linspace(0.05, 1.0, 20), 
                           min_samples_range=np.linspace(100, 500, 20, dtype=int),
                           max_outlier_ratio=0.2):
    
    for min_samples in min_samples_range:
        for eps in eps_range:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embedding)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters == 3:
                non_outliers = labels != -1
                outlier_ratio = 1 - (np.sum(non_outliers) / len(labels))
                
                if outlier_ratio > max_outlier_ratio:
                    continue
                
                filtered_data = original_data[non_outliers]
                filtered_embedding = embedding[non_outliers]
                filtered_labels = labels[non_outliers]
                
                print(f"\nFound 3 clusters with eps={eps:.2f}, min_samples={min_samples}")
                print(f"Outliers removed: {outlier_ratio*100:.2f}% of data")
                
                return filtered_data, filtered_embedding, filtered_labels, non_outliers

    labels = DBSCAN(eps=1.0, min_samples=500).fit_predict(embedding)
    non_outliers = labels != -1
    filtered_data = embedding[non_outliers]
    filtered_labels = labels[non_outliers]
    outlier_ratio = 1 - (np.sum(non_outliers) / len(labels))
    print(f"Fallback: Outliers removed: {outlier_ratio*100:.2f}% of data")
    return filtered_data, filtered_embedding, filtered_labels, non_outliers
