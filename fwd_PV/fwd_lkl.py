from math import pi
import numpy as np
import jax.numpy as jnp
from jax import grad
from .tools.cosmo import z_cos, speed_of_light
from astropy.coordinates import SkyCoord
import astropy.units as u
from jax.config import config
config.update("jax_enable_x64", True)

from fwd_PV.velocity_box import ForwardModelledVelocityBox

EPS = 1e-50

class ForwardLikelihoodBox(ForwardModelledVelocityBox):
    def __init__(self, N_SIDE, L_BOX, PV_data, MB_data, smoothing_scale, window, Pk_type, Pk_data, N_POINTS=501):
        super().__init__(N_SIDE, L_BOX, smoothing_scale, window, Pk_type, Pk_data)
        r_hMpc, e_rhMpc, RA, DEC, z_obs = PV_data
        r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).cartesian.xyz)
        self.r_hat = r_hat
        self.sigmad = e_rhMpc * 100.
        self.RA  = RA
        self.DEC = DEC
        delta_MB, L_BOX_MB, N_GRID_MB = MB_data
        self.los_density = self.get_los_density(delta_MB, L_BOX_MB, N_GRID_MB, N_POINTS)
        r = np.linspace(1., 245., N_POINTS)
        self.r = r.reshape((N_POINTS, 1))
        self.delta_r = np.mean((r[1:] - r[:-1]))
        self.r_hMpc = r_hMpc.reshape((1,-1))
        self.e_rhMpc = e_rhMpc.reshape((1,-1))
        self.z_cos = z_cos(self.r, self.OmegaM)
        self.cz_obs = (speed_of_light * z_obs).reshape((1,-1))
        
    def get_los_density(self, delta_MB, L_BOX_MB, N_GRID_MB, N_POINTS=201):
        r_hat = np.array(SkyCoord(ra=self.RA*u.deg, dec=self.DEC*u.deg).cartesian.xyz)
        r_hat = r_hat.reshape((1,3,-1))
        r = np.linspace(1., 245., N_POINTS)
        r = r.reshape((N_POINTS, 1, 1))
        cartesian_pos = (r * r_hat)
        l  = L_BOX_MB / N_GRID_MB
        MB_indices = ((cartesian_pos + L_BOX_MB / 2.) / l).astype(int)
        self.indices = ((cartesian_pos + self.L_BOX / 2.) / self.l).astype(int)
        delta_los = delta_MB[MB_indices[:,0,:], MB_indices[:,1,:], MB_indices[:,2,:]]
        return delta_los
    
    def log_lkl(self, delta_k, OmegaM, sig_v):
        V_r = self.Vr_grid(delta_k, OmegaM)
        Vr_los = V_r[self.indices[:,0,:], self.indices[:,1,:], self.indices[:,2,:]]
        cz_pred = speed_of_light * self.z_cos + (1. + self.z_cos) * Vr_los
        delta_cz_sigv = (cz_pred - self.cz_obs)/sig_v
        p_r = self.r * self.r * np.exp(-0.5 * ((self.r - self.r_hMpc)/self.e_rhMpc)**2) * (1. + self.los_density)
        p_r_norm = np.trapz(p_r, self.r, axis=0)
        exp_delta_cz = jnp.exp(-0.5*delta_cz_sigv**2)/jnp.sqrt(2 * pi * sig_v**2) 
        p_cz = (jnp.trapz(exp_delta_cz * p_r / p_r_norm, self.r, axis=0))
        lkl_ind = jnp.log(p_cz)
        lkl = jnp.sum(-lkl_ind)
        return lkl

    def grad_lkl(self, delta_k, OmegaM, sig_v):
        lkl_grad = grad(self.log_lkl, 0)(delta_k, OmegaM, sig_v)
        return jnp.array([-lkl_grad[0], lkl_grad[1]])

    def lnprob_Om(self, OmegaM, delta_k, sigma8, sig_v):
        if((OmegaM < 0.1) or (OmegaM > 0.6)):
            return -np.inf
        return -self.log_lkl(delta_k, OmegaM, sig_v) + self.lnprob_s8(sigma8, OmegaM, delta_k)

    def lnprob_sigv(self, sig_v, delta_k, OmegaM, sigma8):
        logP = -self.log_lkl(delta_k, OmegaM, sig_v)
        return logP
