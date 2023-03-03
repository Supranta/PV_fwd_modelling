from math import pi
import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit
from functools import partial
from .tools.cosmo import z_cos, speed_of_light
from astropy.coordinates import SkyCoord
import astropy.units as u
from jax.config import config
config.update("jax_enable_x64", True)

from fwd_PV.velocity_box import ForwardModelledVelocityBox

EPS = 1e-50

class ForwardLikelihoodBox(ForwardModelledVelocityBox):
    def __init__(self, N_SIDE, L_BOX, kh, pk, PV_data, MB_data, coord_system_box, N_POINTS=201):
        super().__init__(N_SIDE, L_BOX, kh, pk)
        r_hMpc, e_rhMpc, RA, DEC, z_obs = PV_data
        delta_MB, L_BOX_MB, N_GRID_MB, coord_system_MB, R_lim = MB_data
        if(coord_system_box=="equatorial"):
            r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).cartesian.xyz)
        elif(coord_system_box=="galactic"):
            print("Using galactic coordinates...")
            r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).galactic.cartesian.xyz)
        elif(coord_system_box=="supergalactic"):
            print("Using supergalactic coordinates...")
            r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).supergalactic.cartesian.xyz)
        self.r_hat = r_hat
        self.sigmad = e_rhMpc * 100.
        self.RA  = RA
        self.DEC = DEC        
        self.R_lim = R_lim
        self.los_density = self.get_los_density(delta_MB, L_BOX_MB, N_GRID_MB, coord_system_MB, N_POINTS)                
        r = np.linspace(1., self.R_lim, N_POINTS)
        self.r = r.reshape((N_POINTS, 1))
        self.delta_r = np.mean((r[1:] - r[:-1]))
        self.r_hMpc = r_hMpc.reshape((1,-1))
        self.e_rhMpc = e_rhMpc.reshape((1,-1))
        self.sigma_mu = 5. / jnp.log(10) * self.e_rhMpc / self.r_hMpc
        self.z_cos = z_cos(self.r, self.OmegaM)
        self.cz_obs = (speed_of_light * z_obs).reshape((1,-1))
        
    def get_los_density(self, delta_MB, L_BOX_MB, N_GRID_MB, coord_system, N_POINTS=2001):
        if(coord_system=="equatorial"):
            r_hat = np.array(SkyCoord(ra=self.RA*u.deg, dec=self.DEC*u.deg).cartesian.xyz)
        elif(coord_system=="galactic"):
            print("Using galactic coordinates for Malquist bias correction...")
            r_hat = np.array(SkyCoord(ra=self.RA*u.deg, dec=self.DEC*u.deg).galactic.cartesian.xyz)
        r_hat = r_hat.reshape((1,3,-1))
        r = np.linspace(1., self.R_lim, N_POINTS)
        r = r.reshape((N_POINTS, 1, 1))
        cartesian_pos_MB = (r * r_hat)
        l  = L_BOX_MB / N_GRID_MB
        MB_indices = ((cartesian_pos_MB + L_BOX_MB / 2.) / l).astype(int)
        delta_los = delta_MB[MB_indices[:,0,:], MB_indices[:,1,:], MB_indices[:,2,:]]
        
        cartesian_pos = (r * self.r_hat)
        self.indices = ((cartesian_pos + self.L_BOX / 2.) / self.l).astype(int)
        return delta_los
    
#     @partial(jit, static_argnums=(0,))
    def get_scale_arr(self, scale):
        scale_arr = jnp.ones(jnp.sum(self.data_len))
        index_end   = jnp.cumsum(self.data_len)
        index_start = jnp.hstack([jnp.array([0]),index_end[:-1]])
        for i in range(self.N_CAT):
            index_slice = jnp.index_exp[index_start[i]:index_end[i]]
            scale_updated = scale[i] * scale_arr[index_slice]
            scale_arr = scale_arr.at[index_slice].set(scale_updated)
        return scale_arr
    
#     @partial(jit, static_argnums=(0,))
    def log_lkl(self, delta_k, scale):
        V_r = self.Vr_grid(delta_k, smooth_R=self.smooth_R)
        Vr_los = V_r[self.indices[:,0,:], self.indices[:,1,:], self.indices[:,2,:]]
        cz_pred = speed_of_light * self.z_cos + (1. + self.z_cos) * Vr_los
        delta_cz_sigv = (cz_pred - self.cz_obs)/self.sig_v
        scale_arr = self.get_scale_arr(scale)
#         p_r = self.r * self.r * jnp.exp(-0.5 * ((self.r - scale_arr * self.r_hMpc)/self.e_rhMpc)**2) * (1. + self.los_density)
#         =========== LOGNORMAL ERROR ==============================
        delta_mu = 5. * jnp.log10(self.r / (scale_arr * self.r_hMpc))
        p_r = self.r * self.r * jnp.exp(-0.5 * delta_mu**2 / self.sigma_mu**2) * (1. + self.los_density)
#         =========== LOGNORMAL ERROR ==============================        
        p_r_norm = jnp.trapz(p_r, self.r, axis=0)
        exp_delta_cz = jnp.exp(-0.5*delta_cz_sigv**2)/jnp.sqrt(2 * pi * self.sig_v**2) 
        p_cz = (jnp.trapz(exp_delta_cz * p_r / p_r_norm, self.r, axis=0))
        lkl_ind = jnp.log(p_cz)
        lkl = jnp.sum(-lkl_ind)
        return lkl

    def log_lkl_scale(self, scale, delta_k):
        return -self.log_lkl(delta_k, scale)   
    
#     @partial(jit, static_argnums=(0,))
    def grad_lkl(self, delta_k, scale=1.):
        # Needs to be updated for the new jax version
        lkl_grad = -grad(self.log_lkl, 0)(delta_k, scale)
        return jnp.array([lkl_grad[0], -lkl_grad[1]])
