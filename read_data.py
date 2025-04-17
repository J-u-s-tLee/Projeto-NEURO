import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import welch

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

def band_power(signal, fs, band, nperseg=2048):

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.trapezoid(psd[idx_band], freqs[idx_band])

    return band_power

def delta_theta_ratio_welch(eeg_signal, fs, nperseg=2048):
    delta_band = (0.5, 4)  
    theta_band = (4, 8)   
    
    delta_power = band_power(eeg_signal, fs, delta_band, nperseg)
    theta_power = band_power(eeg_signal, fs, theta_band, nperseg)
    
    ratio_d_t = delta_power / theta_power
    
    return ratio_d_t

def magnitude_of_movement(data):
    accel_data = data [:, 2:5]
    magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
    return magnitude

def cross_correlation(eeg_signal, movement_signal):

    eeg_signal = eeg_signal - np.mean(eeg_signal)
    movement_signal = movement_signal - np.mean(movement_signal)
    corr = np.correlate(eeg_signal, movement_signal, mode='full')
    lags = np.arange(-len(eeg_signal) + 1, len(eeg_signal))
    return lags, corr

def featureVect(data_dict, fs=1000, window_duration=30):
    fs = 1000
    window_size = window_duration * fs  
    features_list = []

    for key, data in data_dict.items():
        n_samples = data.shape[0]
        n_windows = n_samples // window_size

        print(f"Processing {key} - {n_windows} windows")

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            
            win_frontal = data[start:end, 0]
            win_parietal = data[start:end, 1]
            win_acc_y = data[start:end, 2]
            win_acc_z = data[start:end, 3]
            win_acc_x = data[start:end, 4]

            delta_f = band_power(win_frontal, fs, (0.5, 4))
            theta_f = band_power(win_frontal, fs, (4, 8))
            sigma_f = band_power(win_frontal, fs, (12, 16))
            beta_f = band_power(win_frontal, fs, (16, 30)) 
            delta_p = band_power(win_parietal, fs, (0.5, 4))
            theta_p = band_power(win_parietal, fs, (4, 8))
            beta_f = band_power(win_frontal, fs, (16, 30))   
            sigma_p = band_power(win_parietal, fs, (12, 16)) 
            beta_p = band_power(win_parietal, fs, (16, 30)) 

            ratio_d_t_f = delta_theta_ratio_welch(win_frontal, fs)
            ratio_d_t_p = delta_theta_ratio_welch(win_parietal, fs)

            var_acc_x = np.var(win_acc_x)
            var_acc_y = np.var(win_acc_y)
            var_acc_z = np.var(win_acc_z)

            magnitude = magnitude_of_movement(data[start:end, :])

            #lags, corr_frontal = cross_correlation(win_frontal, magnitude)

            #lags, corr_parietal = cross_correlation(win_parietal, magnitude)

            feature_names = [
                 "delta_f", "theta_f", "sigma_f", "beta_f", 
                 "delta_p", "theta_p", "sigma_p", "beta_p",
                 "ratio_d_t_f", "ratio_d_t_p",
                 "var_acc_x", "var_acc_y", "var_acc_z",
                 "mean_magnitude", "max_magnitude", "min_magnitude"]
                 #"mean_corr_frontal", "mean_corr_parietal"]
            
            features = [
                delta_f, theta_f, delta_p, theta_p, 
                sigma_f, beta_f, sigma_p, beta_p,
                ratio_d_t_f, ratio_d_t_p,
                var_acc_x, var_acc_y, var_acc_z,
                np.mean(magnitude),
                np.max(magnitude), 
                np.min(magnitude),
                #np.mean(corr_frontal), 
                #np.mean(corr_parietal)  
            ]
            
            features_list.append(dict(zip(feature_names, features)))

    return features_list
