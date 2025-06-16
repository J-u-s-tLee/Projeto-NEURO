from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import matplotlib.pyplot as plt

def Standardization(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df)
    else:
        scaled_array = scaler.transform(df)
    
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns)
    return scaled_df, scaler

import matplotlib.pyplot as plt
import os

def Histogram_plot(df, df_scaled, feature_name, output_path="unsupervised_output/hist_single_feature.png"):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(df[feature_name], bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Original - {feature_name}')
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel('Frequency')

    axes[1].hist(df_scaled[feature_name], bins=20, color='salmon', edgecolor='black')
    axes[1].set_title(f'Scaled - {feature_name}')
    axes[1].set_xlabel(f'{feature_name} (scaled)')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
