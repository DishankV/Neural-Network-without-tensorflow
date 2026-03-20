import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # Prevent log(0)
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))