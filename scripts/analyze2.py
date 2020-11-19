import h5py as h5
import matplotlib.pyplot as plt
from matplotlib import cm

from fwd_PV.io import process_config_analysis

import sys

configfile = sys.argv[1]

N_GRID, L_BOX, savedir, _, _ = process_config_analysis(configfile)

with h5.File(savedir+'/mean_std.h5','r') as f:
    delta_mean = f['delta_mean'][:]
    delta_std  = f['delta_std'][:]
    Vr_mean    = f['Vr_mean'][:]
    Vr_std     = f['Vr_std'][:]

extent = (-L_BOX, L_BOX, -L_BOX, L_BOX)

fig, ax = plt.subplots(2,3,figsize=(15,10))
for i in range(2):
    for j in range(3):
        ax[i,j].plot(0., 0., 'kx')
ax[0,0].imshow(Vr_mean[N_GRID//2,:,:], cmap=cm.coolwarm, extent=extent, vmin=-400, vmax=400)
ax[0,1].imshow(Vr_mean[:,N_GRID//2,:], cmap=cm.coolwarm, extent=extent, vmin=-400, vmax=400)
ax[0,2].imshow(Vr_mean[:,:,N_GRID//2], cmap=cm.coolwarm, extent=extent, vmin=-400, vmax=400)

ax[1,0].imshow(Vr_std[N_GRID//2,:,:], cmap=cm.coolwarm, extent=extent, vmin=0, vmax=200)
ax[1,1].imshow(Vr_std[:,N_GRID//2,:], cmap=cm.coolwarm, extent=extent, vmin=0, vmax=200)
ax[1,2].imshow(Vr_std[:,:,N_GRID//2], cmap=cm.coolwarm, extent=extent, vmin=0, vmax=200)

plt.tight_layout()
plt.savefig(savedir+'/figs/Vr_mean_std.png', dpi=150)
plt.close()

fig, ax = plt.subplots(2,3,figsize=(15,10))

for i in range(2):
    for j in range(3):
        ax[i,j].plot(0., 0., 'kx')

ax[0,0].imshow(delta_mean[N_GRID//2,:,:], cmap=cm.coolwarm, extent=extent, vmin=-2, vmax=2)
ax[0,1].imshow(delta_mean[:,N_GRID//2,:], cmap=cm.coolwarm, extent=extent, vmin=-2, vmax=2)
ax[0,2].imshow(delta_mean[:,:,N_GRID//2], cmap=cm.coolwarm, extent=extent, vmin=-2, vmax=2)

ax[1,0].imshow(delta_std[N_GRID//2,:,:], cmap=cm.coolwarm, extent=extent, vmin=0, vmax=2)
ax[1,1].imshow(delta_std[:,N_GRID//2,:], cmap=cm.coolwarm, extent=extent, vmin=0, vmax=2)
ax[1,2].imshow(delta_std[:,:,N_GRID//2], cmap=cm.coolwarm, extent=extent, vmin=0, vmax=2)

plt.tight_layout()
plt.savefig(savedir+'/figs/delta_mean_std.png', dpi=150)
plt.close()
