import numpy as np
import camb
from camb import model, initialpower

speed_of_light = 299792

def z_cos(r_hMpc, Omega_m):
    Omega_L = 1. - Omega_m
    q0 = Omega_m/2.0 - Omega_L
    return (1.0 - np.sqrt(1 - 2*r_hMpc*100*(1 + q0)/speed_of_light))/(1.0 + q0)

def r2dL(r, OmegaM):
    z_cos_arr = z_cos(r, OmegaM)
    return r * (1 + z_cos_arr)

def r2mu(r):
    return 5 * np.log10(r) + 25

def mu2r(mu):
    return 10**((mu - 25.)/5.)

def camb_PS():
    print("Calculating CAMB power spectrum....")
    pars = camb.CAMBparams()
    b_frac = 0.022 / 0.122
    h = 0.7
    OmegaM = 0.27
    pars.set_cosmology(H0=100*h, ombh2=b_frac * OmegaM * h**2, omch2=(1. - b_frac) * OmegaM * h**2)
    pars.InitPower.set_params(ns=0.965)
    
    pars.set_matter_power(redshifts=[0.], kmax=5.0)

    results = camb.get_results(pars)

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=5e-3, maxkh=5., npoints = 2000)
    
    return kh, pk[0]

def get_Pk(sigma8, OmegaM, s8_arr, Om_arr, Pk_arr):
    Om_min = np.min(Om_arr)
    Om_max = np.max(Om_arr)
    
    delta_Om = (Om_arr[1:] - Om_arr[:-1])[0]
    
    #if(OmegaM > Om_max):
    #    return Pk_arr[-1]
    
    x = (OmegaM - Om_min)/delta_Om

    i = int(x)
    s = x%1
    
    sigma_fid = (1-s) * s8_arr[i] + s * s8_arr[i+1]
    Pk = (1-s) * Pk_arr[i] + s * Pk_arr[i+1]

    #print(i, s)
    #print(sigma8, sigma_fid)    
    #print(np.mean(Pk / Pk_arr[i]))
    return (sigma8 / sigma_fid)**2 * Pk 
