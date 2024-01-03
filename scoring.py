import numpy as np

def calculate_confmatrix(predict, real_label):
    """
    label list:
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 4,
        'Sleep stage R': 5,
        'Sleep stage ?': 6
    """
    n = len(predict)
    cfm = np.zeros((6, 6), dtype=int)
    for i in range(n):
        cfm[real_label[i], predict[i]] += 1
    return cfm

def mf1_score(cfm):
    pre = 0
    rec = 0
    for i in range(6):
        TP = cfm[i,i]
        FP = np.sum(cfm[:,i]) - TP
        FN = np.sum(cfm[i,:]) - TP
        pre += TP / (TP + FP)
        rec += TP / (TP + FN)
        print(i,TP,FP,FN,pre,rec)
    pre /= 6
    rec /= 6
    return 2 * pre * rec / (pre + rec)

def acc_score(cfm):
    return np.sum(cfm * np.eye(6)) / np.sum(cfm)

def kappa_score(cfm):
    po = acc_score(cfm)
    pe = 0
    for i in range(6):
        pe += np.dot(cfm[:,i], cfm[i,:])
    pe /= np.sum(cfm) ** 2
    return (po - pe) / (1 - pe)

def score(predict, real_label, method):
    """
    3 methods can be applied: 
        MF1 : Macro F1
        ACC : overall accuracy
        kappa : Cohen's kappa
    predict and real_label should both be 1-D numpy matrix with same length
    """
    cfm = calculate_confmatrix(predict, real_label)
    if method == 'MF1':
        return mf1_score(cfm)
    if method == 'ACC':
        return acc_score(cfm)
    if method == 'kappa':
        return kappa_score(cfm)