import numpy as np


def calculate_dobule_ema(data, alpha1=0.5, alpha2=0.3, windows_size=3):
    ema1 = np.zeros(len(data))
    ema2 = np.zeros(len(data))

    ema1[0] = data[0]
    for t in range(1, len(data)):
        ema1[t] = alpha1 * data[t] + (1 - alpha1) * ema1[t - 1]

    ema2[0] = ema1[0]
    for t in range(1, len(data)):
        ema2[t] = alpha2 * ema1[t] + (1 - alpha2) * ema2[t - 1]

    dema = 2 * ema1 - ema2
    return dema
