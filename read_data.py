import os
import numpy as np
import h5py

def readData(diretorio, num_files):
    base_path = os.path.dirname(__file__)
    directory = os.path.join(base_path, diretorio)

    data_dict = {}

    for i in range(1, num_files + 1):
        file_path = os.path.join(directory, f'continuous{i}.mat')

        if file_path.endswith('.mat'):
            print(f"Reading file: {file_path}")
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data_bin'][:]
                    data = np.array(data, dtype=np.float32)
                    data_dict[f'continuous{i}'] = data
                    data_dict[f'continuous{i}'] = np.delete(data_dict[f'continuous{i}'], 0, axis=1)
            except Exception as e:
                print(f"Error reading file (h5py): {file_path}")
                print(str(e))
    
    return data_dict
