import read_data
import signal_processing
import data_processing
import umap_reduction
import pandas as pd

data_dict = read_data.readData('Continuous\Continuous', 15)
processed_dict = signal_processing.FilterSignals(data_dict, fs=1000) 
#signal_processing.plot_signals(data_dict, processed_dict, key='continuous1', fs=1000, duration=10)
features = read_data.featureVect(processed_dict)
df_features = pd.DataFrame(features)
Data = data_processing.Standardization(df_features)
embedding = umap_reduction.dimensionality_reduce(Data)
labels = umap_reduction.dbscan_clustering(embedding, eps=0.3, min_samples=10)
filtered_embedding, filtered_labels = umap_reduction.remove_outliers(embedding, labels)
umap_reduction.plot_embedding(filtered_embedding, labels=filtered_labels)

k_means = kmeans.Kmeans_method(embedding, 3)
k_means_embedding, k_means_labels = umap_reduction.remove_outliers(embedding, k_means)

mask = np.isin(k_means, k_means_labels)
filtered_data = Data[mask]
filtered_labels = k_means_labels
Labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=8)
Labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=9)
Labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=13)
