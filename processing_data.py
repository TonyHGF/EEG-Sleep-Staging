import numpy as np
def make_diff(signal):
    res = []
    for i in range(len(signal)-1):
        res.append(signal[i+1]-signal[i])
    res.append(0)
    return res

def list2np(raw):
    x = np.concatenate(raw)
    diff_raw = []
    for raw_signal in raw:
        diff_raw.append(make_diff(raw_signal))
        
    x_diff = np.concatenate(diff_raw)
    x = np.concatenate([x, x_diff])
    return x

from scipy.signal import butter, lfilter

def butter_bandpass_filter(sig, lowcut, highcut, Fs, order=7):
    nf = 0.5 * Fs
    low = lowcut / nf
    high = highcut / nf
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, sig)
    return y

def signal_filter(sig):
    return butter_bandpass_filter(sig, 0.5, 30, 100)

def down_sampling(sig):
    # print(len(sig))
    signal = []
    for i in range(0, len(sig), 10):
        signal.append(sig[i])
    # print(signal)
    return signal

def generate_matrix(split_signals):
    res = []
    labels = []
    for (label, signals) in split_signals:
        labels.append(label)
        for i in range(len(signals)):
            signals[i] = down_sampling(signals[i])
        res.append(list2np(signals))
        
    X = np.vstack(res)
    Y = np.vstack(labels)
    return X, Y