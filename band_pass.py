from scipy.signal import butter, lfilter

def butter_bandpass_filter(sig, lowcut, highcut, Fs, order=7):
    nf = 0.5 * Fs
    low = lowcut / nf
    high = highcut / nf
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, sig)
    return y

def signal_filter(sig):
    return butter_bandpass_filter(sig, 0.4, 30, 100)