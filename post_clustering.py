import os
import pandas as pd
from scipy.io import loadmat
import h5py
import numpy as np

def relabel_dataset(input_folder, output_folder, label_maps):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            
            df = pd.read_csv(filepath)
            
            mouse_name = None
            for mouse in label_maps.keys():
                if filename.startswith(mouse):
                    mouse_name = mouse
                    break
            
            if mouse_name is None:
                print(f"Mouse name not found for file {filename}, skipping.")
                continue
            
            label_map = label_maps[mouse_name]
            
            df['class'] = df['class'].map(label_map)
            
            output_path = os.path.join(output_folder, filename)
            df.to_csv(output_path, index=False)
            print(f"Saved relabeled file to {output_path}")

def combine_and_remap_classes(output_folder, reverse_map, combined_filename):
    all_dfs = []
    for filename in os.listdir(output_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(output_folder, filename)
            df = pd.read_csv(filepath)
            all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['class'] = combined_df['class'].map(reverse_map)

    combined_df.to_csv(combined_filename, index=False)
    print(f"\nFinal combined file saved to {combined_filename}")

def merge_mat_files(input_files, output_file, var_name=None):
    if os.path.exists(output_file):
        return
    
    arrays = []

    for file in input_files:
        mat_data = loadmat(file)
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if var_name is None:
            key = keys[0]
        else:
            key = var_name
        
        arrays.append(mat_data[key])

    combined_data = np.vstack(arrays)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('combined_data', data=combined_data, compression='gzip')

    print(f"Data saved at '{output_file}'")
