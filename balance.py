import itertools
import numpy as np

from sklearn.cluster import KMeans

def get_balance_class_downsample(x, y):
    """
    Balance the number of samples of all classes by (downsampling):
        1. Find the class that has a smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        idx = np.random.permutation(idx)[:n_min_classes]
        balance_x.append(x[idx])
        balance_y.append(y[idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y

def generate_prototypes(X, y):
    """
    使用K-means算法进行原型生成，聚类数量与少数类样本数量相匹配。
    
    参数:
    X -- 原始数据集的特征矩阵。
    y -- 数据集的标签向量。
    
    返回:
    X_new -- 经过原型生成处理后的新特征矩阵。
    y_new -- 新特征矩阵对应的标签向量。
    """
    y = y.astype(int)
    # 分离多数类和少数类
    majority_class = np.argmax(np.bincount(y))
    minority_class = 1 - majority_class

    X_majority = X[y == majority_class]
    X_minority = X[y == minority_class]

    # 确定聚类数量为少数类样本的数量
    n_clusters = len(X_minority)

    # 对多数类应用K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_majority)
    prototypes = kmeans.cluster_centers_

    # 创建新的数据集
    X_new = np.vstack((prototypes, X_minority))
    y_new = np.array([majority_class] * n_clusters + [minority_class] * n_clusters)

    return X_new, y_new