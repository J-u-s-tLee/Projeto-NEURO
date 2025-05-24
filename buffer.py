import numpy as np
import h5py

def read_samples_from_mat(file_path, dataset_name='data_bin', num_channels=6):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name]
        n_samples = data.shape[0]

        for i in range(n_samples):
            sample = data[i][1:]
            yield sample

def stream_30s_windows(file_paths, fs=1000, window_sec=30, num_channels=5):
    window_size = fs * window_sec
    buffer = []

    for file_path in file_paths:
        print(f"Reading {file_path}")
        sample_generator = read_samples_from_mat(file_path, num_channels=num_channels + 1)
        for sample in sample_generator:
            buffer.append(sample)
            if len(buffer) == window_size:
                window = np.array(buffer)
                yield window
                buffer = []
