import numpy as np

def delete_question_mark(X, y):
    N = len(X)
    pos = np.where(y != 6)
    return X[pos], y[pos]

def split_wake_and_asleep(X, y):
    y_new = y.copy()
    pos = np.where(y_new != 0)
    y_new[pos] = 1
    return X, y_new

def split_eye_movement(X, y):
    pos = np.where(y != 0)
    X_new = X[pos]
    y_new = y[pos]
    pos = np.where(y_new != 5)
    y_new[pos] = 0
    pos = np.where(y_new == 5)
    y_new[pos] = 1
    return X_new, y_new

def split_deep_sleep(X, y):
    pos = np.where((y != 0) * (y != 5))
    X_new = X[pos]
    y_new = y[pos]
    pos = np.where((y_new == 1) ^ (y_new == 2))
    y_new[pos] = 0
    pos = np.where((y_new == 3) ^ (y_new == 4))
    y_new[pos] = 1
    return X_new, y_new

def split_1_2(X, y):
    pos = np.where((y == 1) ^ (y == 2))
    X_new = X[pos]
    y_new = y[pos]
    pos = np.where(y_new == 1)
    y_new[pos] = 0
    pos = np.where(y_new == 2)
    y_new[pos] = 1
    return X_new, y_new

def split_3_4(X, y):
    pos = np.where((y == 3) ^ (y == 4))
    X_new = X[pos]
    y_new = y[pos]
    pos = np.where(y_new == 3)
    y_new[pos] = 0
    pos = np.where(y_new == 4)
    y_new[pos] = 1
    return X_new, y_new