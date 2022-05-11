import numpy as np
from matplotlib import pyplot as plt

res = np.load("pres_c4_c5_c6_120MeV_100ofthem_Jung2011.npz")
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(res['C4+'][:, 0], res['C4+'][:, 1], s=1.5, color='black', label=r'$C^{4+}$')
plt.scatter(res['C5+'][:, 0], res['C5+'][:, 1], s=1.5, color='blue', label=r'$C^{5+}$')
plt.scatter(res['C6+'][:, 0], res['C6+'][:, 1], s=1.5, color='deeppink', label=r'$C^{6+}$')
#plt.legend()
plt.xlabel("Deflection along x axis [m]")
plt.ylabel("Deflection along y axis [m]")
ax.text(0.7, 0.7, r'$C^{4+}, C^{5+}, C^{6+}$' + "\n \nOwn code", fontsize=12, color='Red', transform=ax.transAxes)
plt.savefig("100C4100C5100C6_120MeV_12MeVstd_Jung2011.pdf", bbox_inches='tight')