import numpy as np
from scipy import stats

data = [99,86,87,88,111,86,103,87,94,78,77,85,86]
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", stats.mode(data))
print("Standard Deviation:", np.std(data))
print("Variance:", np.var(data))
print("Range:", np.max(data) - np.min(data))
