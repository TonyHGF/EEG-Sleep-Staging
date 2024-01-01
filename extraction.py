import numpy as np
from scipy import stats
import os
from frequency_domain_operation import *

def calculate_zero_crossings(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)

# Hjorth mobility

def calculate_hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(signal, n=2)
    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return mobility, complexity

def higuchi_fractal_dimension(signal, kmax=10): # by GPT
    """
    计算Higuchi Fractal Dimension。

    参数:
    signal : np.array
        一维信号数组。
    kmax : int
        最大的k值。

    返回:
    HFD : float
        Higuchi Fractal Dimension的值。
    """
    n = len(signal)
    lk = np.zeros(kmax)
    x = np.arange(1, kmax + 1)
    y = np.zeros(kmax)
    for k in range(1, kmax + 1):
        lm = np.zeros((k,))
        for m in range(1, k + 1):
            ll = 0
            for i in range(1, int(np.floor((n - m) / k)) + 1):
                ll += np.abs(signal[m + i * k - 1] - signal[m + (i - 1) * k - 1])
            lm[m - 1] = ll * n / (k * int(np.floor((n - m) / k)) * (n - 1))
        lk[k - 1] = np.log(np.mean(lm))
        y[k - 1] = np.log(1. / k)
    hfd, _ = np.polyfit(y, lk, 1)
    return hfd

def petrosian_fractal_dimension(signal):
    """
    计算Petrosian Fractal Dimension。

    参数:
    signal : np.array
        一维信号数组。

    返回:
    PFD : float
        Petrosian Fractal Dimension的值。
    """
    n = len(signal)
    # 计算差分序列
    diff_signal = np.diff(signal)
    # 计算零交叉数
    N_delta = np.sum(diff_signal[:-1] * diff_signal[1:] < 0)
    pfd = np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))
    return pfd

def signal_extraction(signal, cond = 1):
    li = []
    # std, iqr, skewness, kurtosis
    li.append(np.std(signal))
    q75, q25 = np.percentile(signal, [75 ,25])
    iqr = q75 - q25
    li.append(iqr)
    li.append(stats.skew(signal))
    li.append(stats.kurtosis(signal))

    # number of zero-crossings
    li.append(calculate_zero_crossings(signal))

    # Hjorth mobility, Hjorth complexity
    mobility, complexity = calculate_hjorth_parameters(signal)
    li.append(mobility)
    li.append(complexity)

    # higuch fractal dimension, petrosian fractal dimension
    hfd = higuchi_fractal_dimension(signal)
    li.append(hfd)
    pfd = petrosian_fractal_dimension(signal)
    li.append(pfd)

    # permutation entropy, binned entropy (4)
    time_entropy_bins = [5, 10, 30, 60]
    for bin in time_entropy_bins:
        li.append(calculate_binned_entropy(signal, bin))
    
    frequency_signals = perform_fft(signal, 100)
    res = calculate_frequency_domain_features(frequency_signals, 100)
    li.append(res['spectral_centroid'])
    li.append(res['spectral_variance'])
    li.append(res['spectral_skewness'])
    li.append(res['spectral_kurtosis'])
    for entro in res['binned_entropies']:
        li.append(entro)
    for value in res['band_powers'].values():
        li.append(value)
    if cond != 1:
        return np.array(li)
    for value in res['power_ratios'].values():
        li.append(value)
    
    return np.array(li)

from read_edf import *
from read_result import *
from split_signal import *
from tqdm import tqdm
import time

label_dir = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 4,
    'Sleep stage R': 5,
    'Sleep stage ?': 6
}

def extract_single_file(edf_path, result_path):
    start_times, durations, labels = read_result_file(result_path)
    res = split_edf_by_annotations(edf_path, start_times, durations, labels)
    split_signals = split_edf_by_30s(res)
    X_list = []
    Y_list = []
    filename_without_extension = os.path.splitext(os.path.basename(edf_path))[0]
    file_name = filename_without_extension.split('-')[0]
    for label_name, sub_signal in tqdm(split_signals, desc=file_name):
        Y_list.append(label_dir[label_name])
        vec_list = []
        for one_of_three in sub_signal:
           vec_list.append(signal_extraction(one_of_three)) 
        vec = np.concatenate(vec_list)
        X_list.append(vec)
    X = np.array(X_list)
    Y = np.array(Y_list)
    print("done:", file_name)
    directory_path = 'result/'
    np.savez(directory_path + file_name, features = X, labels = Y)