import data_splits
import numpy as np
import pandas as pd
import SVM

X_train, X_val, X_test, y_train, y_val, y_test = data_splits.load_and_split_data()

param_grid = {
   'kernel': ['linear', 'rbf'],
   'C': [0.1, 1, 10],
   'gamma': ['scale', 'auto'],
}

best_model = SVM.gridsearch(X_train, X_val, y_train, y_val, param_grid)
SVM.svm_test(best_model, X_test, y_test)
