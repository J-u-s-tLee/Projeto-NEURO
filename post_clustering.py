import os
import pandas as pd

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
