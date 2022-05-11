from matplotlib import pyplot as plt
import numpy as np    
from matplotlib.pyplot import cm


with open(".txt", "r") as f:
    len_of_final_coords_at_detectorscreen = f.readline()
    names = []
    for i in range(len_of_final_coords_at_detectorscreen):
        names.append( f.readline() )
    title_of_graph = f.readline()
    E = f.readline()
    B = f.readline()
    lengthss = f.readline()
    z_det = f.readline()
    y_electrode_bottom = f.readline()
    tols = f.readline()

colors = iter(cm.rainbow(np.linspace(0,1, 2 * len_of_final_coords_at_detectorscreen))) # if you have many chunks of particles (many species), this helps select 1 DIFFERENT color to represent each chunk. 

plt.figure()
res = np.load('{}.npz'.format(title_of_graph))
for key in names: # for each chunk
    print(key)
    c = next(colors)
    c = np.reshape(c, (1, c.shape[0]) )
    plt.scatter(res[key][:, 0], res[key][:, 1], s=0.2, label=key, c=c)
plt.xlabel("Deflection along x axis [meters]")
plt.ylabel("Deflection along y axis [meters]")
# plt.title("")
plt.legend()
plt.savefig(".pdf".format(title_of_graph), bbox_inches='tight')
