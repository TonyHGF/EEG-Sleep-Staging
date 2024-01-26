from tqdm import tqdm
from tqdm import trange
import numpy as np

def int_model(X, y, W_model, REM_model, N_model, model_12, model_34):
    len = X.shape[0]
    # Y_pred = np.zeros(len)
    Y_pred = []
    for i in trange(len):
        x = X[i:]
        y_pred = W_model.predict(x)
        if y_pred[0] == 0:
            Y_pred.append(0)
            continue
        y_pred = REM_model.predict(x)
        if y_pred[0] == 1:
            Y_pred.append(5)
            continue
        y_pred = N_model.predcit(x)
        if y_pred[0] == 0:
            y_pred = model_12.predict(x)
            if y_pred[0] == 0:
                Y_pred.append(1)
            else:
                Y_pred.append(2)
        else:
            y_pred = model_34.predict(x)
            if y_pred[0] == 0:
                Y_pred.append(3)
            else:
                Y_pred.append(4)            
    return np.array(Y_pred)