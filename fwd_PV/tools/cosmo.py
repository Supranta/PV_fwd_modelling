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
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    
    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    results = camb.get_results(pars)

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=1e-2, maxkh=5., npoints = 2000)
    
    return kh, pk[0]

def Pk_interpolated(s8, Om, s8_mesh, Om_mesh, Pk_arr, i_mesh, j_mesh, factor):
    dist = np.sqrt((Om_mesh - Om)**2 + factor * (s8_mesh - s8)**2)
    dist_min = np.sort(dist.flatten())[:4]
    weights = (1./dist_min)
    weights = weights/np.sum(weights)
    min_inds = np.argsort(dist.flatten())[:4]
    
    Om_mins = Om_mesh.flatten()[min_inds]
    sig8_mins = s8_mesh.flatten()[min_inds]
    i_ind = i_mesh.flatten()[min_inds]
    j_ind = j_mesh.flatten()[min_inds]
    
    Pk_near = Pk_arr[i_ind, j_ind]
    Pk_interp = np.sum(weights.reshape(4,1) * Pk_near, axis=0)
    
    return Pk_interp
