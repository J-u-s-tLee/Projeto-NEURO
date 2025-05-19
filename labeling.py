import matplotlib.pyplot as plt
import numpy as np

def ordering(samples, labels, start_time, end_time, feature_index=0):
    samples = np.array(samples)
    labels = np.array(labels)
    samples_per_minute = 2
    start_idx = int(start_time * samples_per_minute)
    end_idx = int(end_time * samples_per_minute)
    end_idx = min(end_idx, len(samples))

    samples = samples[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    time_axis = np.arange(len(samples)) * 0.5

    label_colors = {0: 'blue', 1: 'red', 2: 'yellow'}

    plt.figure(figsize=(12, 6))

    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            time_axis[mask],
            samples[mask, feature_index],
            color=label_colors[label],
            label=f'Label {label}',
            s=50,
            alpha=0.8
        )

    plt.xlabel('Time (minutes)')
    plt.ylabel(f'Feature {feature_index}')
    plt.title(f'Feature {feature_index} vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

