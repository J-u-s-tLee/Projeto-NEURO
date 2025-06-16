from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def BandPass(order=4, Wn=[0.5, 100], fs=1000):
    sos = signal.butter(order, Wn, btype='bandpass', fs=fs, output='sos')
    return sos

def Notch(w0=50, Q=30, fs=1000):
    b, a = signal.iirnotch(w0, Q, fs)
    return b, a

def LowPass(order=4, Wn=50, fs=1000):
    sos = signal.butter(order, Wn, btype='low', fs=fs, output='sos')
    return sos

def FilterSignals(data_dict, fs=1000):
    bandpass_sos = BandPass(order=4, Wn=[0.5, 100], fs=fs)
    b_notch, a_notch = Notch(w0=50, Q=30, fs=fs)
    lowpass_sos = LowPass(order=4, Wn=5, fs=fs)

    filtered_dict = {}

    for key, data in data_dict.items():
        filtered_data = np.zeros_like(data)

        for ch in [0, 1]:
            bandpassed = signal.sosfilt(bandpass_sos, data[:, ch])
            notch_filtered = signal.lfilter(b_notch, a_notch, bandpassed)
            filtered_data[:, ch] = notch_filtered

        for ch in [2, 3, 4]:
            filtered_data[:, ch] = signal.sosfilt(lowpass_sos, data[:, ch])

        filtered_dict[key] = filtered_data

    return filtered_dict

def plot_signals(original_dict, filtered_dict, key='continuous1', fs=1000, duration=20, save_path=None):

    channel_names = ['EEG Frontal', 'EEG Parietal', 'Acc X', 'Acc Y', 'Acc Z']
    time = np.arange(0, duration, 1/fs)

    original = original_dict[key][:len(time), :]
    filtered = filtered_dict[key][:len(time), :]

    fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Signals before and after processing ({key})", fontsize=16)

    for i in range(5):
        axes[i, 0].plot(time, original[:, i], color='deepskyblue')
        axes[i, 0].set_ylabel(channel_names[i])
        axes[i, 0].set_title("Original")

        axes[i, 1].plot(time, filtered[:, i], color='tomato')
        axes[i, 1].set_title("Filtered")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
