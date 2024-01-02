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
    
def plot_signals(signal_labels, signals, sample_frequency, save=False):
    num_channels = len(signal_labels)
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels))
    for i in range(num_channels):
        total_duration = len(signals[i]) / sample_frequency
        time_axis = np.linspace(0, total_duration, len(signals[i]))
        # print("#", total_duration)
        axes[i].plot(time_axis, signals[i], label=signal_labels[i])
        axes[i].set_ylabel("Amplitude")
        axes[i].legend()
        axes[i].set_xlabel("Time (seconds)")
        axes[i].set_xlim(0, total_duration)
    plt.subplots_adjust(hspace=0.5)
    if save:
        plt.savefig('pre\plot.png')
    plt.show()