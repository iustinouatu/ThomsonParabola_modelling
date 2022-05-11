import numpy as np
from matplotlib import pyplot as plt

res = np.load("pres_c4_c5_c6_120MeV_100ofthem_Jung2011.npz")
fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(res['Xe52+'][:, 0], res['Xe52+'][:, 1], s=1.5, color='black', label=r'$Xe^{52+}$')
plt.scatter(res['Xe53+'][:, 0], res['Xe53+'][:, 1], s=1.5, color='blue', label=r'$Xe^{53+}$')
plt.scatter(res['Xe54+'][:, 0], res['Xe54+'][:, 1], s=1.5, color='deeppink', label=r'$Xe^{54+}$')
#plt.legend()
plt.xlabel("Deflection along x axis [m]")
plt.ylabel("Deflection along y axis [m]")
ax.text(0.7, 0.7, "Xe52+, Xe53+, Xe54+" + "\n \nOwn code", fontsize=12, color='Red', transform=ax.transAxes)
plt.savefig("100Xe52100Xe53100Xe54_1310MeV_131MeVstd_Jung2011.pdf", bbox_inches='tight')