import numpy as np
import camb
from camb import model, initialpower
import matplotlib.pyplot as plt
import h5py as h5
import time

import sys, os
sys.path.append(os.path.abspath('..'))

from fwd_PV.io import config_Pk

configfile = sys.argv[1]

def get_sigma8(OmegaM, As):
    pars = camb.CAMBparams()

    baryon_frac = 0.022 / (0.022 + 0.122)
    
    H0 = 67.5
    h = H0/100.
    
    pars.set_cosmology(H0=H0, ombh2=baryon_frac * OmegaM * h**2, omch2=(1. - baryon_frac) * OmegaM * h**2)
    pars.InitPower.set_params(As, ns=0.965)

    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    results = camb.get_results(pars)

#     pars.NonLinear = model.NonLinear_none
    
    return results.get_sigma8()[0]

def camb_PS(OmegaM, As):
    print("Calculating CAMB power spectrum....")
    pars = camb.CAMBparams()
    H0 = 67.5
    h = H0/100.
    baryon_frac = 0.022 / (0.022 + 0.122)
    
    pars.set_cosmology(H0=H0, ombh2=baryon_frac * OmegaM * h**2, omch2=(1. - baryon_frac) * OmegaM * h**2)
    pars.InitPower.set_params(As=As, ns=0.965)
    
    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    results = camb.get_results(pars)

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=1e-2, maxkh=5., npoints = 2000)
    
    return kh, pk[0]

savedir, Om_min, Om_max, As_min, As_max, Om_size, As_size = config_Pk(configfile)

As_min = As_min * 1e-9
As_max = As_max * 1e-9

OmegaM_arr = np.linspace(Om_min, Om_max, Om_size)
As_arr = np.linspace(As_min, As_max, As_size)

Pk_arr = np.zeros((Om_size, As_size, 2000))
sigma8 = np.zeros((Om_size, As_size))

i_mesh = np.zeros((Om_size, As_size)).astype(int)
j_mesh = np.zeros((Om_size, As_size)).astype(int)

start_time = time.time()
for i in range(Om_size):
    for j in range(As_size):
        print(i,j)
        t0 = time.time()
        As = As_arr[j]*(0.315/OmegaM_arr[i])
        sigma8[i,j] = get_sigma8(OmegaM_arr[i], As)
        kh, pk = camb_PS(OmegaM_arr[i], As)
        
        Pk_arr[i,j] = pk
        
        i_mesh[i,j] = i
        j_mesh[i,j] = j
        t1 = time.time()
        print("Time per step: %2.1f"%(t1 - t0))
end_time = time.time()

print("Time taken: %2.2f seconds"%(end_time - start_time))
Om_mesh = np.tile(OmegaM_arr.reshape((Om_size,1)), (1,As_size))
As_mesh = np.tile(As_arr.reshape((1,As_size)), (Om_size,1))

factor = ((Om_max - Om_min)/(np.max(sigma8, axis=1) - np.min(sigma8)))**2

with h5.File('../'+savedir+'/Pk_arr.h5', 'w') as f:
     f['Om_mesh'] = Om_mesh
     f['As_mesh'] = As_mesh
     f['s8_mesh'] = sigma8
     f['i_mesh']  = i_mesh
     f['j_mesh']  = j_mesh
     f['Pk_arr']  = Pk_arr
     f['factor']  = factor
