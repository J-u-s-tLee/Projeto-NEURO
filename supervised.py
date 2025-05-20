import data_splits
import RF
import SVM
import numpy as np
import pandas as pd
import os

# Load and split data
X_train, X_val, X_test, y_train, y_val, y_test = data_splits.load_and_split_data()

X_test = pd.concat([X_val, X_test], axis=0).reset_index(drop=True)
y_test = pd.concat([y_val, y_test], axis=0).reset_index(drop=True)

os.makedirs('supervised_output', exist_ok=True)

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
