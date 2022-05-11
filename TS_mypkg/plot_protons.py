import numpy as np
from matplotlib import pyplot as plt

res = np.load("100protons_10MeV_1MeVstd_Jung2011.npz")
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(res['proton'][:, 0], res['proton'][:, 1], s=1.5)
plt.xlabel("Deflection along x axis [m]")
plt.ylabel("Deflection along y axis [m]")
ax.text(0.7, 0.7, "Protons" + "\n \nOwn code", fontsize=12, color='Red', transform=ax.transAxes)
plt.savefig("100protons_10MeV_1MeVstd_Jung2011.pdf", bbox_inches='tight')