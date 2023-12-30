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

def generate_matrix(split_signals):
    res = []
    label = []
    for i in range(len(split_signals)):
        label.append(split_signals[i][0])
        res.append(list2np(split_signals[i][1]))
        
    X = np.vstack(res)
    Y = np.vstack(label)

    return X, Y