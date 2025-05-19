import read_data
import feature_extraction
import signal_processing
import data_processing
import umap_reduction
import dbscan
import pandas as pd
import kmeans
import labeling
import silhouette
import os
import numpy as np

data_dict = read_data.readData('Continuous', 15)

processed_dict = signal_processing.FilterSignals(data_dict, fs=1000) 
features = feature_extraction.featureVect(processed_dict)
df_features = pd.DataFrame(features)
Data = data_processing.Standardization(df_features)
embedding = umap_reduction.dimensionality_reduce(Data)

dbscan_labels = dbscan.dbscan_find_3_clusters(embedding)
filtered_embedding, filtered_labels = dbscan.remove_outliers(embedding, dbscan_labels)
dbscan.plot_embedding(filtered_embedding, labels=filtered_labels)

kmeans_labels = kmeans.Kmeans_method(filtered_embedding, 3)
dbscan.plot_embedding(filtered_embedding, labels=kmeans_labels)

silhouette.silhouette_plot(filtered_embedding, kmeans_labels, n_clusters=3)
silhouette.silhouette_plot(filtered_embedding, filtered_labels, n_clusters=3)

print(f"K-means Silhouette Score: {silhouette.silhouettescore(filtered_embedding, kmeans_labels)}")
print(f"DBSCAN Silhouette Score: {silhouette.silhouettescore(filtered_embedding, filtered_labels)}")

mask = np.isin(dbscan_labels, filtered_labels)
filtered_data = Data[mask]

labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=8)
labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=9)
labeling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=13)

df_output = filtered_data.copy()
df_output['class'] = filtered_labels

os.makedirs('supervised_output', exist_ok=True)

df_output.to_csv('supervised_output/labelled_data.csv', index=False)
