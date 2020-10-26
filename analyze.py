import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from fwd_PV.tools.fft import Fourier_ks
from fwd_PV.tools.cosmo import camb_PS
from math import pi

savedir = 'fwd_PV_runs/sample_prior'
N_BOX = 64
L = 400.
l = L / N_BOX
V = L**3

kh, pk = camb_PS()

_, k_abs = Fourier_ks(N_BOX, l)

k_bins = np.logspace(np.log10(pi / L / 2.), np.log10(2*pi * N_BOX / L), 51)
k_bincentre = np.sqrt(k_bins[1:]*k_bins[:-1])

def measure_Pk(delta_k, k_norm, k_bins):
    Pk_sample = []
    for i in range(len(k_bins)-1):
        select_k = (k_norm > k_bins[i])&(k_norm < k_bins[i+1])
        if(np.sum(select_k) < 1):
            Pk_sample.append(0.)
        else:
            Pk_sample.append(np.mean(np.abs(delta_k[select_k]))**2 * V)
    return np.array(Pk_sample)

Pk_measured_list = []

J = np.complex(0., 1.)

for i in range(200, 600):
    print("Reading mcmc_"+str(i)+".h5")
    f = h5.File(savedir + '/mcmc_'+str(i)+'.h5', 'r')
    delta_k = f['delta_k'][:]
    delta_k_complex = delta_k[0] + J*delta_k[1]
    Pk_sample = measure_Pk(delta_k_complex, k_abs, k_bins)
    Pk_measured_list.append(Pk_sample)

Pk_measured = np.array(Pk_measured_list)

# np.save(savedir+'/Pk_measured.npy', Pk_measured)
# np.save(savedir+'/k_bins.npy', k_bins)

Pk_measured_mean = np.mean(Pk_measured, axis=0)
Pk_measured_low  = np.percentile(Pk_measured, 16., axis=0)
Pk_measured_high = np.percentile(Pk_measured, 84., axis=0)

plt.loglog(k_bincentre, Pk_measured_mean, color='b')
plt.fill_between(k_bincentre, Pk_measured_low, Pk_measured_high, color='b', alpha=0.3)
plt.loglog(kh, pk, 'k', label='CAMB')
plt.savefig(savedir+'/figs/Pk_samples.png', dpi=150)

print("Pk_measured.shape: "+str(Pk_measured.shape))
