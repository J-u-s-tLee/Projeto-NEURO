import read_data
import feature_extraction
import signal_processing
import data_processing
import umap_reduction
import dbscan
import pandas as pd
import kmeans
import labelling
import silhouette
import os
import numpy as np

# Change according to dataset location
data_dict = read_data.readData('Continuous\Continuous', 15)

os.makedirs('unsupervised_output', exist_ok=True)

processed_dict = signal_processing.FilterSignals(data_dict, fs=1000) 
features = feature_extraction.featureVect(processed_dict)
df_features = pd.DataFrame(features)
Data = data_processing.Standardization(df_features)
embedding = umap_reduction.dimensionality_reduce(Data)

dbscan_labels = dbscan.dbscan_find_3_clusters(embedding)
filtered_embedding, filtered_labels = dbscan.remove_outliers(embedding, dbscan_labels)
umap_reduction.plot_embedding(filtered_embedding, labels=filtered_labels, save_path='unsupervised_output/dbscan.png')

kmeans_labels = kmeans.Kmeans_method(filtered_embedding, 3)
umap_reduction.plot_embedding(filtered_embedding, labels=kmeans_labels, save_path='supervised_output/kmeans.png')

silhouette.silhouette_plot(filtered_embedding, kmeans_labels, n_clusters=3, save_path='unsupervised_output/silhouette_plot_kmeans.png')
silhouette.silhouette_plot(filtered_embedding, filtered_labels, n_clusters=3, save_path='unsupervised_output/silhouette_plot_dbscan.png')

print(f"\nK-means Silhouette Score: {silhouette.silhouettescore(filtered_embedding, kmeans_labels)}")
print(f"DBSCAN Silhouette Score: {silhouette.silhouettescore(filtered_embedding, filtered_labels)}")

mask = np.isin(dbscan_labels, filtered_labels)
filtered_data = Data[mask]

labelling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=8, save_path='unsupervised_output/labelling_ratio_d_t_f.png')
labelling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=9, save_path='unsupervised_output/labelling_ratio_d_t_p.png')
labelling.ordering(filtered_data, labels=filtered_labels, start_time=20, end_time=200, feature_index=13, save_path='unsupervised_output/labelling_mean_magnitude.png')

df_output = filtered_data.copy()
df_output['class'] = filtered_labels

df_output.to_csv('unsupervised_output/labelled_data.csv', index=False)
