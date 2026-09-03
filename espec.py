import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

fig, axs = plt.subplots(1, 2)

for filename in sys.argv[1:]:
	label = Path(filename).resolve().parts[-2]
	k, E, Ec = np.loadtxt(filename, unpack=True)
	axs[0].plot(k, E, label=label)
	axs[1].plot(k, Ec)
	
axs[0].plot(k, k**(-5./3.))

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylim((1.0e-5, 1.0e1))
axs[0].legend(loc='best')

axs[1].plot(k, np.full_like(k, 1.5))
axs[1].plot(k, np.full_like(k, 1.7))
axs[1].set_xscale('log')
plt.show()
