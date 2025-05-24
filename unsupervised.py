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
import data_splits
import joblib

mouse_id = {
    'Mouse1': [1, 15],
    'Mouse2': [16, 19],
    'Mouse3': [20, 23]
}

os.makedirs('data_dict', exist_ok=True)
os.makedirs('unsupervised_output', exist_ok=True)

silhouette_scores = []

for name, (start, end) in mouse_id.items():
    print(f"\n--- Processing {name} ({start} to {end}) ---")

    data_dict = read_data.readData('Continuous', start, end, 'data_dict')

    processed_dict = signal_processing.FilterSignals(data_dict, fs=1000)
    
    features = feature_extraction.featureVect(processed_dict)
    df_features = pd.DataFrame(features)
    Data, scaler = data_processing.Standardization(df_features)
    joblib.dump(scaler, r'unsupervised_output\scaler.joblib')

    embedding = umap_reduction.dimensionality_reduce(Data)

    filtered_data, filtered_embedding, dbscan_labels, non_outliers = dbscan.dbscan_find_3_clusters(Data, embedding)
    umap_reduction.plot_embedding(filtered_embedding, labels=dbscan_labels, save_path=f'unsupervised_output/{name}_dbscan.png')

    kmeans_labels = kmeans.Kmeans_method(filtered_embedding, 3)
    umap_reduction.plot_embedding(filtered_embedding, labels=kmeans_labels, save_path=f'unsupervised_output/{name}_kmeans.png')

    silhouette.silhouette_plot(filtered_embedding, kmeans_labels, n_clusters=3, save_path=f'unsupervised_output/{name}_silhouette_kmeans.png')
    silhouette.silhouette_plot(filtered_embedding, dbscan_labels, n_clusters=3, save_path=f'unsupervised_output/{name}_silhouette_dbscan.png')

    score_kmeans = silhouette.silhouettescore(filtered_embedding, kmeans_labels)
    score_dbscan = silhouette.silhouettescore(filtered_embedding, dbscan_labels)

    silhouette_scores.append({
        'mouse_id': name,
        'kmeans_score': float(score_kmeans),
        'dbscan_score': float(score_dbscan)
    })

    silhouette.save_silhouette_scores(silhouette_scores, output_dir='unsupervised_output')

    labelling.ordering(filtered_data, labels=dbscan_labels, start_time=0, end_time=600, feature_index=8, save_path=f'unsupervised_output/{name}_dbscan_labelling_ratio_d_t_f.png')
    labelling.ordering(filtered_data, labels=dbscan_labels, start_time=0, end_time=600, feature_index=9, save_path=f'unsupervised_output/{name}_dbscan_labelling_ratio_d_t_p.png')
    labelling.ordering(filtered_data, labels=dbscan_labels, start_time=0, end_time=600, feature_index=13, save_path=f'unsupervised_output/{name}_dbscan_labelling_mean_magnitude.png')

    labelling.ordering(filtered_data, labels=kmeans_labels, start_time=0, end_time=600, feature_index=8, save_path=f'unsupervised_output/{name}_kmeans_labelling_ratio_d_t_f.png')
    labelling.ordering(filtered_data, labels=kmeans_labels, start_time=0, end_time=600, feature_index=9, save_path=f'unsupervised_output/{name}_kmeans_labelling_ratio_d_t_p.png')
    labelling.ordering(filtered_data, labels=kmeans_labels, start_time=0, end_time=600, feature_index=13, save_path=f'unsupervised_output/{name}_kmeans_labelling_mean_magnitude.png')

    df_output = filtered_data.copy()
    df_output['class'] = dbscan_labels
    df_output.to_csv(f'unsupervised_output/{name}_labelled_data.csv', index=False)

    combined_dict = read_data.combine_channels(data_dict)
    Data_test = data_splits.get_original_samples(combined_dict, non_outliers, fs=1000, window_sec=30)
    data_splits.save_mat(Data_test, 'unsupervised_output', f'{name}.mat')
