import numpy as np

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