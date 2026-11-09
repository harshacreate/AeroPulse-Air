import numpy as np

def chronological_split(n, test_size=0.25):
    if n < 8 or not 0.1 <= test_size <= 0.5:
        raise ValueError("need at least 8 rows and test_size between 0.1 and 0.5")
    cut = int(n*(1-test_size))
    return np.arange(cut), np.arange(cut,n)
