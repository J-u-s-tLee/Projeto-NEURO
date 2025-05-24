import os
import numpy as np
import h5py

def readData(diretory, start_file, end_file, output_dir):
    base_path = os.path.dirname(__file__)
    directory = os.path.join(base_path, diretory)
    output_path = os.path.join(output_dir, f'data_cache_{start_file}_{end_file}.npz')

    if os.path.exists(output_path):
        print(f"Loading cached data from {output_path} ...")
        return dict(np.load(output_path, allow_pickle=True))

    data_dict = {}

    for i in range(start_file, end_file + 1):
        file_path = os.path.join(directory, f'continuous{i}.mat')

        if file_path.endswith('.mat'):
            print(f"Reading file: {file_path}")
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data_bin'][:]
                    data = np.array(data, dtype=np.float32)
                    data = np.delete(data, 0, axis=1) 
                    data_dict[f'continuous{i}'] = data
            except Exception as e:
                print(f"Error reading file (h5py): {file_path}")
                print(str(e))

    np.savez(output_path, **data_dict)
    print(f"\nData cached to {output_path}")
    return data_dict

def combine_channels(data_dict):
    combined_data = []
    keys = sorted(data_dict.keys(), key=lambda x: int(x.replace('continuous', '')))
    
    for key in keys:
        channel_data = data_dict[key].squeeze()
        combined_data.append(channel_data)
    
    combined_array = np.concatenate(combined_data)
    return combined_array
