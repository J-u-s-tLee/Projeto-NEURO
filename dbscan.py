import numpy as np 
from sklearn.cluster import DBSCAN

def dbscan_find_3_clusters(embedding, 
                           eps_range=np.linspace(0.05, 1.0, 5), 
                           min_samples_range=[70, 90, 110]):
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
