import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import savemat
import os
import numpy as np

def load_and_split_data(filepath='supervised_output/final_labelled_data.csv', label_column='class', train_size=0.7, random_state=42):
    df = pd.read_csv(filepath)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=False)

    X = df.drop(columns=[label_column, 'index'])
    y = df[label_column]
    indices = df['index']

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=(1.0 - train_size), random_state=random_state)

    return X_train, X_test, y_train, y_test, idx_train, idx_test


def save_mat(data, directory, file_name):
    os.makedirs(directory, exist_ok=True)
    
    if isinstance(data, pd.DataFrame):
        data_to_save = data.to_numpy()
    elif isinstance(data, np.ndarray):
        data_to_save = data
    else:
        raise ValueError("Data must be a pandas DataFrame or numpy ndarray")
    
    key_name = os.path.splitext(file_name)[0]
    filepath = os.path.join(directory, file_name)
    savemat(filepath, {key_name: data_to_save})

def get_original_samples(original_data, feature_indices, fs=1000, window_sec=30):
    window_size = fs * window_sec
    original_samples = []
    
    for idx in feature_indices:
        start = idx * window_size
        end = (idx + 1) * window_size
        original_samples.append(original_data[start:end])
    
    return np.concatenate(original_samples, axis=0)
