import h5py as h5
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

import fwd_PV.io as io
from fwd_PV.velocity_box import ForwardModelledVelocityBox
from fwd_PV.tools.cosmo import camb_PS

configfile = 'sample_config.txt'

N_GRID, L_BOX, likelihood, coord_system_box = io.config_box(configfile)
savedir, N_SAVE, N_RESTART                  = io.config_io(configfile)

kh, pk = camb_PS()

all_f = os.listdir(savedir)
all_f = [int(f[5:-3]) for f in all_f if f.startswith('mcmc_')]
fname = savedir + '/mcmc_'+str(max(all_f))+'.h5'

VelocityBox = ForwardModelledVelocityBox(N_GRID, L_BOX, kh, pk)

with h5.File(fname, 'r') as f:
    delta_k = jnp.array(f['delta_k'])
    delta_x = VelocityBox.get_delta_grid(delta_k, smooth_R = VelocityBox.smooth_R)
    print(delta_x.shape)
    
fig, axs = plt.subplots(1, 3, figsize=(10,4))
axs[0].pcolor(delta_x[64,:,:])
axs[1].pcolor(delta_x[:,64,:])
axs[2].pcolor(delta_x[:,:,64])
fig.tight_layout()
plt.show()
