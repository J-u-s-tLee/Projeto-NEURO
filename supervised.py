import data_splits
import post_clustering
import RF
import SVM
import numpy as np
import pandas as pd
import os
import h5py
import joblib

os.makedirs('supervised_output', exist_ok=True)

label_maps = {
    'Mouse1': {0:'w', 1:'nrem', 2:'rem'},
    'Mouse2': {0:'nrem', 1:'rem', 2:'w'},
    'Mouse3': {0:'nrem', 1:'rem', 2:'w'}
}
reverse_map = {'w': 0, 'nrem': 1, 'rem': 2}

combined_filename = os.path.join('supervised_output', 'final_labelled_data.csv')

if not os.path.exists(combined_filename):
    post_clustering.relabel_dataset('unsupervised_output', 'supervised_output', label_maps)
    post_clustering.combine_and_remap_classes('supervised_output', reverse_map, combined_filename)

post_clustering.merge_mat_files([r'unsupervised_output\Mouse1.mat', r'unsupervised_output\Mouse2.mat', r'unsupervised_output\Mouse3.mat',], r'supervised_output\combined_all_mice.h5')

X_train, X_test, y_train, y_test, idx_train, idx_test = data_splits.load_and_split_data()
idx_test = idx_test.to_numpy()

with h5py.File(r'supervised_output\combined_all_mice.h5', 'r') as f:
    Data = f['combined_data'][:]

Data_test = data_splits.get_original_samples(Data, idx_test, fs=1000, window_sec=30)
data_splits.save_mat(pd.DataFrame(Data_test), 'supervised_output', 'Data_test.mat')

param_grid = {
    'n_estimators': [10, 25, 50],  
    'max_depth': [10, 20, None], 
    'min_samples_split': [2, 5, 10]  
}

best_params = RF.grid_search(X_train, y_train, param_grid)
best_rf = RF.train_model(X_train, y_train, best_params)
results = RF.evaluate_model(best_rf, X_test, y_test, output_path='supervised_output')

param_grid = {
   'kernel': ['linear', 'rbf'],
   'C': [0.1, 1, 10],
   'gamma': ['scale', 'auto'],
}

best_svm = SVM.gridsearch(X_train, y_train, param_grid)
SVM.svm_test(best_svm, X_test, y_test, output_path='supervised_output')

data_splits.save_mat(X_test, 'supervised_output', 'online_samples.mat')

joblib.dump(best_svm, r'supervised_output\svm_model.joblib')
