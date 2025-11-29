import numpy as np

def calculate_dobule_ema(data, alpha1=0.5, alpha2=0.3, windows_size = 3):
    
    """
    Calculate Double Exponential Moving Average (DEMA) for the given data.
    
    Parameters:
    data (list or np.array): Input data series.
    alpha1 (float): Smoothing factor for the first EMA.
    alpha2 (float): Smoothing factor for the second EMA.
    
    Returns:
    np.array: DEMA values.
    """

    ema1 = np.zeros(len(data))
    ema2 = np.zeros(len(data))
    
    ema1[0] = data[0]
    for t in range(1, len(data)):
        ema1[t] = alpha1 * data[t] + (1 - alpha1) * ema1[t-1]
    
    ema2[0] = ema1[0]
    for t in range(1, len(data)):
        ema2[t] = alpha2 * ema1[t] + (1 - alpha2) * ema2[t-1]
    
    dema = 2 * ema1 - ema2
    return dema

if __name__ == "__main__":
    incomes = [10000, 100] 
    ewa_incomes = calculate_dobule_ema(incomes)
    print("Original Incomes: ", incomes)
    print("Double EMA Incomes: ", ewa_incomes)