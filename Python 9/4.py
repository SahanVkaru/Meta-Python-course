import matplotlib.pyplot as plt
import numpy as np

x = np.random.uniform(0, 10, 1000)
plt.hist(x, bins=5)
plt.title("Uniform Distribution")
plt.show()