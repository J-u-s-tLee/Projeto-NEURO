import data_splits
import RF
import numpy as np
import pandas as pd

# Load and split data
X_train, X_val, X_test, y_train, y_val, y_test = data_splits.load_and_split_data()
