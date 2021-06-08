from matplotlib import pyplot as mp
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

x_values = np.linspace(-3, 3, 120)
for mu, sig in [(0 , 1)]:
    mp.plot(x_values, gaussian(x_values, mu, sig))

mp.show()