import data_splits
import post_clustering
import RF
import SVM
import numpy as np
import pandas as pd
import os

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

X_train, X_val, X_test, y_train, y_val, y_test = data_splits.load_and_split_data()

X_train = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_train = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)


param_grid = {
    'n_estimators': [10, 25, 50],  
    'max_depth': [10, 20, None], 
    'min_samples_split': [2, 5, 10]  
}

best_params = RF.grid_search(X_train, y_train, param_grid)
best_model = RF.train_model(X_train, y_train, best_params)
results = RF.evaluate_model(best_model, X_test, y_test, output_path='supervised_output')

param_grid = {
   'kernel': ['linear', 'rbf'],
   'C': [0.1, 1, 10],
   'gamma': ['scale', 'auto'],
}

best_model = SVM.gridsearch(X_train, X_val, y_train, y_val, param_grid)
SVM.svm_test(best_model, X_test, y_test, output_path='supervised_output')

data_splits.save_Xtest(X_test, 'supervised_output')
