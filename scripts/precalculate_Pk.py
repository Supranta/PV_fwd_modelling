import numpy as np
import camb
from camb import model, initialpower
import matplotlib.pyplot as plt
import h5py as h5
import time

import sys, os
sys.path.append(os.path.abspath('..'))

from fwd_PV.io import config_Pk

h = 0.675
OmegaM = 0.315
ns = 0.965
baryon_frac = 0.022 / 0.122
kmax = 5.
kmin = 5e-3

configfile = sys.argv[1]

def camb_PS(OmegaM, As):
    print("Calculating CAMB power spectrum....")
    pars = camb.CAMBparams()
    
    pars.set_cosmology(H0=100.*h, ombh2=baryon_frac * OmegaM * h**2, omch2=(1. - baryon_frac) * OmegaM * h**2)
    pars.InitPower.set_params(As=As, ns=ns)
    
    pars.set_matter_power(redshifts=[0.], kmax=kmax)
    results = camb.get_results(pars)

    sigma8 = results.get_sigma8()[0]

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = 5000)
    
    return kh, pk[0], sigma8

savedir, Om_min, Om_max, Om_size = config_Pk(configfile)

As_0 = 2.1e-9

OmegaM_arr = np.linspace(Om_min, Om_max, Om_size)

Pk_arr = np.zeros((Om_size, 5000))
sigma8_arr = np.zeros((Om_size))

start_time = time.time()
for i in range(Om_size):
    print(i)
    t0 = time.time()
    As = As_0*(OmegaM/OmegaM_arr[i])
    #sigma8_arr[i] = get_sigma8(OmegaM_arr[i], As)
    kh, pk, sigma8_arr[i] = camb_PS(OmegaM_arr[i], As)
        
    Pk_arr[i] = pk
        
    t1 = time.time()
    print("Time per step: %2.2f"%(t1 - t0))
end_time = time.time()

print("Time taken: %2.2f seconds"%(end_time - start_time))

with h5.File('../'+savedir+'/Pk_arr.h5', 'w') as f:
    f['Om_arr'] = OmegaM_arr
    f['s8_arr'] = sigma8_arr
    f['Pk_arr'] = Pk_arr
    f['kh']     = kh
    
