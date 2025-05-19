import read_data
import feature_extraction
import signal_processing
import data_processing
import umap_reduction
import dbscan
import pandas as pd

data_dict = read_data.readData('Continuous\Continuous', 15)
processed_dict = signal_processing.FilterSignals(data_dict, fs=1000) 
features = feature_extraction.featureVect(processed_dict)
df_features = pd.DataFrame(features)
Data = data_processing.Standardization(df_features)
embedding = umap_reduction.dimensionality_reduce(Data)
labels = dbscan.dbscan_find_3_clusters(embedding)
filtered_embedding, filtered_labels = dbscan.remove_outliers(embedding, labels)
dbscan.plot_embedding(filtered_embedding, labels=filtered_labels)

k_means = kmeans.Kmeans_method(embedding, 3)
k_means_embedding, k_means_labels = umap_reduction.remove_outliers(embedding, k_means)

mask = np.isin(k_means, k_means_labels)
filtered_data = Data[mask]
filtered_labels = k_means_labels
labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=8)
labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=9)
labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=13)
