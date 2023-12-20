import pyedflib
import matplotlib.pyplot as plt
import numpy as np

def read_edf_file(file_path):
    try:
        f = pyedflib.EdfReader(file_path)
        num_signals = f.signals_in_file
        signal_labels = f.getSignalLabels()
        signals = [f.readSignal(i) for i in range(num_signals)]
        sample_frequency = f.getSampleFrequency(0)
        f.close()
        return signal_labels, signals, sample_frequency

    except Exception as e:
        print(f"Error reading .edf file: {e}")
        return None
    
def plot_signals(signal_labels, signals, sample_frequency):
    num_channels = len(signal_labels)
    fig, axes = plt.subplots(num_channels, 1, sharex=True, figsize=(10, 2 * num_channels))
    for i in range(num_channels):
        total_duration = len(signals[i]) / sample_frequency
        time_axis = np.linspace(0, total_duration, len(signals[i]))
        print(time_axis.shape, signals[i].shape)
        axes[i].plot(time_axis, signals[i], label=signal_labels[i])
        axes[i].set_ylabel("Amplitude")
        axes[i].legend()
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()