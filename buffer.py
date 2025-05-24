import numpy as np
import time
from scipy.io import loadmat

def read_samples_from_mat(file_path, dataset_name='Data_test'):
    mat_data = loadmat(file_path)
    data = mat_data[dataset_name]
    for sample in data:
        yield sample


def stream_30s_windows(file_path, fs=1000, window_sec=30):
    window_size = fs * window_sec
    buffer = []
    buffer_start_time = None

    for sample in read_samples_from_mat(file_path):
        if buffer_start_time is None:
            buffer_start_time = time.time()

        buffer.append(sample)

        if len(buffer) == window_size:
            elapsed_time = time.time() - buffer_start_time
            remaining_time = max(0, window_sec - elapsed_time)
            if remaining_time > 0:
                time.sleep(remaining_time)

            window_array = np.array(buffer)
            yield {'data': window_array}

            buffer = []
            buffer_start_time = None

