import numpy as np
import matplotlib.pyplot as plt
import sys

k, E, Ec = np.loadtxt(sys.argv[1], unpack=True)
fig, axs = plt.subplots(1, 2)
axs[0].plot(k, E)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylim((1.0e-5, 1.0e1))
axs[1].plot(k, Ec)
axs[1].set_xscale('log')
plt.show()
