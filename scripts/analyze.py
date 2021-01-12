import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys, os
sys.path.append(os.path.abspath('..'))
from fwd_PV.tools.fft import Fourier_ks
from fwd_PV.tools.cosmo import camb_PS
from fwd_PV.io import process_config_analysis
from math import pi

configfile = sys.argv[1]

N_BOX, L, _, savedir, PROCESS_3D_V_DELTA, CALCULATE_MEAN_STD, _, _ = process_config_analysis(configfile)

try:
    PLOT_LKL = bool(sys.argv[4] == '1')
except:
    PLOT_LKL = True
try:
    PLOT_Pk  = bool(sys.argv[5] == '1')
except:
    PLOT_Pk = True

J = np.complex(0., 1.)

def measure_Pk(delta_k, k_norm, k_bins):
    Pk_sample = []
    for i in range(len(k_bins)-1):
        select_k = (k_norm > k_bins[i])&(k_norm < k_bins[i+1])
        if(np.sum(select_k) < 1):
            Pk_sample.append(0.)
        else:
            Pk_sample.append(np.mean(delta_k[0,select_k]**2 + delta_k[1,select_k]**2) * V)
    return np.array(Pk_sample)

N_START = int(sys.argv[2])
N_END = int(sys.argv[3])

savedir = '../'+savedir

if(PLOT_LKL):
    ln_prob_list = []
    for i in range(N_START, N_END):
        print("Reading mcmc_"+str(i)+".h5")
        f = h5.File(savedir + '/mcmc_'+str(i)+'.h5', 'r')
        ln_prob = f['ln_prob'].value
        ln_prob_list.append(ln_prob)
    plt.plot(np.arange(N_START, N_END), np.array(ln_prob_list))
    plt.savefig(savedir+'/figs/ln_prob.png', dpi=150)
    plt.close()

if(PLOT_Pk):
    l = L / N_BOX
    V = L**3

    kh, pk = camb_PS()

    _, k_abs = Fourier_ks(N_BOX, l)

    k_bins = np.logspace(np.log10(pi / L), np.log10(2*pi * N_BOX / L), 31)
    k_bincentre = np.sqrt(k_bins[1:]*k_bins[:-1])

    Pk_measured_list = []

    for i in range(N_START, N_END):
        print("Reading mcmc_"+str(i)+".h5")
        f = h5.File(savedir + '/mcmc_'+str(i)+'.h5', 'r')
        delta_k = f['delta_k'][:]
        if(i>N_START):
            Pk_sample = measure_Pk(delta_k, k_abs, k_bins)
            Pk_measured_list.append(Pk_sample)
    
    Pk_measured = np.array(Pk_measured_list)
    np.save(savedir+'/Pk_samples.npy', Pk_measured)
    Pk_measured_mean = np.mean(Pk_measured, axis=0)
    Pk_measured_low  = np.percentile(Pk_measured, 16., axis=0)
    Pk_measured_high = np.percentile(Pk_measured, 84., axis=0)

    plt.ylim(3.0e+2, 4.0e+4)
    plt.xlim(5.e-3, 1.)
    plt.loglog(k_bincentre, Pk_measured_mean, color='b')
    plt.fill_between(k_bincentre, Pk_measured_low, Pk_measured_high, color='b', alpha=0.3)
    plt.loglog(kh, pk, 'k', label='CAMB')
    plt.legend()
    plt.savefig(savedir+'/figs/Pk_samples.png', dpi=150)
    plt.close()
