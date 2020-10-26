import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from fwd_PV.tools.fft import Fourier_ks
from fwd_PV.tools.cosmo import camb_PS
from math import pi

savedir = 'fwd_PV_runs/sample_prior'
N_BOX = 64
L = 500.
l = L / N_BOX
V = L**3

kh, pk = camb_PS()

_, k_abs = Fourier_ks(N_BOX, l)

k_bins = np.logspace(np.log10(pi / L / 2.), np.log10(2*pi * N_BOX / L), 41)
k_bincentre = np.sqrt(k_bins[1:]*k_bins[:-1])

J = np.complex(0., 1.)

def measure_Pk(delta_k, k_norm, k_bins):
    delta_k_complex = delta_k[0] + J * delta_k[1]
    Pk_sample = []
    for i in range(len(k_bins)-1):
        select_k = (k_norm > k_bins[i])&(k_norm < k_bins[i+1])
        if(np.sum(select_k) < 1):
            Pk_sample.append(0.)
        else:
            Pk_sample.append(np.mean(np.abs(delta_k_complex[select_k]))**2 * V)
    return np.array(Pk_sample)

def measure_phase(delta_k, k_norm, k_bins):
    delta_k_x = delta_k[0]
    delta_k_y = delta_k[1]
    
    delta_k_abs = np.sqrt(delta_k_x**2 + delta_k_y**2)

    phase_x_sample = []
    phase_y_sample = []
    for i in range(len(k_bins)-1):
        select_k = (k_norm > k_bins[i])&(k_norm < k_bins[i+1])
        if(np.sum(select_k) < 1):
            phase_x_sample.append(0.)
            phase_y_sample.append(0.)
        else:
            delta_x_select = delta_k_x[select_k]
            delta_y_select = delta_k_y[select_k]
            delta_select = delta_k_abs[select_k]
            phase_x_sample.append(np.mean(delta_x_select / delta_select))
            phase_y_sample.append(np.mean(delta_y_select / delta_select))
    return np.array(phase_x_sample), np.array(phase_y_sample)


Pk_measured_list = []
phase1_list = []
phase2_list = []

for i in range(200, 400):
    print("Reading mcmc_"+str(i)+".h5")
    f = h5.File(savedir + '/mcmc_'+str(i)+'.h5', 'r')
    delta_k = f['delta_k'][:]
    Pk_sample = measure_Pk(delta_k, k_abs, k_bins)
    Pk_measured_list.append(Pk_sample)
    phase1, phase2 = measure_phase(delta_k, k_abs, k_bins)
    phase1_list.append(phase1)
    phase2_list.append(phase2)
Pk_measured = np.array(Pk_measured_list)

# np.save(savedir+'/Pk_measured.npy', Pk_measured)
# np.save(savedir+'/k_bins.npy', k_bins)

Pk_measured_mean = np.mean(Pk_measured, axis=0)
Pk_measured_low  = np.percentile(Pk_measured, 16., axis=0)
Pk_measured_high = np.percentile(Pk_measured, 84., axis=0)

plt.semilogx(k_bincentre, Pk_measured_mean, color='b')
plt.fill_between(k_bincentre, Pk_measured_low, Pk_measured_high, color='b', alpha=0.3)
plt.axhline(1000., color='k')
#plt.loglog(kh, pk, 'k', label='CAMB')
plt.savefig(savedir+'/figs/Pk_samples.png', dpi=150)
plt.close()

phase1_arr = np.array(phase1_list)
phase2_arr = np.array(phase2_list)

phase1_mean = np.mean(phase1_arr, axis=0)
phase2_mean = np.mean(phase2_arr, axis=0)
phase1_low = np.percentile(phase1_arr, 16., axis=0)
phase2_low = np.percentile(phase2_arr, 16., axis=0)
phase1_high = np.percentile(phase1_arr, 84., axis=0)
phase2_high = np.percentile(phase2_arr, 84., axis=0)

plt.semilogx(k_bincentre, phase1_mean, color='r', label='Phase1')
plt.semilogx(k_bincentre, phase2_mean, color='b', label='Phase2')
plt.fill_between(k_bincentre, phase1_low, phase1_high, alpha=0.3, color='r')
plt.fill_between(k_bincentre, phase2_low, phase2_high, alpha=0.3, color='b')
plt.savefig(savedir+'/figs/phases.png', dpi=150)

print("Pk_measured.shape: "+str(Pk_measured.shape))
