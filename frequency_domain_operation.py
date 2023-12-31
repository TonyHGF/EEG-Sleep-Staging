import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

def perform_fft(signal, fs):
    fft_result = fft(signal)
    n = len(signal)
    freq = fftfreq(n, d=1/fs)
    return fft_result

def calculate_binned_entropy(signal, num_bins): # by GPT
    # 将信号数据分割成箱
    histogram, bin_edges = np.histogram(signal, bins=num_bins, density=True)
    
    # 计算每个箱的概率
    probabilities = histogram * np.diff(bin_edges)

    # 移除概率为0的箱以防止在计算对数时出现问题
    probabilities = probabilities[probabilities > 0]

    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_frequency_domain_features(signal, fs):
    # 执行傅里叶变换并获取频率分量
    n = len(signal)
    freq = fftfreq(n, d=1/fs)
    fft_result = fft(signal)
    psd = np.abs(fft_result) ** 2

    # 频域统计量
    spectral_centroid = np.sum(freq * psd) / np.sum(psd)
    spectral_variance = np.sum(((freq - spectral_centroid) ** 2) * psd) / np.sum(psd)
    spectral_skewness = skew(np.abs(fft_result))
    spectral_kurtosis = kurtosis(np.abs(fft_result))

    # 分箱傅立叶熵
    bin_sizes = [2, 3, 5, 10, 30, 60, 100]
    binned_entropies = [calculate_binned_entropy(psd, bin_size) for bin_size in bin_sizes]

    # 频带功率
    bands = {'delta_slow': (0.4, 1), 'delta_fast': (1, 4), 'theta': (4, 8),
             'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30)}
    band_powers = {band: np.sum(psd[(freq >= low) & (freq <= high)]) for band, (low, high) in bands.items()}

    # 频带比例功率
    power_ratios = {
        'fast_delta_theta': band_powers['delta_fast'] + band_powers['theta'],
        'alpha_theta': band_powers['alpha'] / band_powers['theta'],
        'delta_beta': band_powers['delta_slow'] / band_powers['beta'],
        'delta_sigma': band_powers['delta_slow'] / band_powers['sigma'],
        'delta_theta': band_powers['delta_slow'] / band_powers['theta']
    }

    # 组合所有特征
    features = {
        'spectral_centroid': spectral_centroid,
        'spectral_variance': spectral_variance,
        'spectral_skewness': spectral_skewness,
        'spectral_kurtosis': spectral_kurtosis,
        'binned_entropies': binned_entropies,
        'band_powers': band_powers,
        'power_ratios': power_ratios
    }

    return features
