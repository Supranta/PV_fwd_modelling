import numpy as np
import jax.numpy as jnp
from jax import grad
from .tools.cosmo import z_cos, speed_of_light
from astropy.coordinates import SkyCoord
import astropy.units as u
from jax.config import config
config.update("jax_enable_x64", True)

from fwd_PV.velocity_box import ForwardModelledVelocityBox

class ChiSquared(ForwardModelledVelocityBox):
    def __init__(self, N_SIDE, L_BOX, kh, pk, r_hMpc, e_rhMpc, RA, DEC, z_obs, interpolate=False):
        super().__init__(N_SIDE, L_BOX, kh, pk)
        r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).cartesian.xyz)
        self.r_hat = r_hat
        self.sigmad = e_rhMpc * 100.
        print("Mean sigma_d: %2.4f"%(np.mean(e_rhMpc * 100)))
        self.cz_obs = speed_of_light * z_obs
        self.cartesian_pos = r_hMpc * r_hat
        self.z_cos = z_cos(r_hMpc, self.OmegaM)
        self.indices = ((self.cartesian_pos +  self.L_BOX/2.) / self.l).astype(int)
        self.sig_v = 150.

    def log_lkl(self, delta_k, A):
        V_r = A * self.Vr_grid(delta_k)
        V_r_tracers = V_r[self.indices[0], self.indices[1], self.indices[2]]
        cz_pred = speed_of_light * self.z_cos + V_r_tracers * (1. + self.z_cos)
        sigma_tot_sq = self.sig_v**2 + self.sigmad**2
        lkl = jnp.sum(0.5 * (self.cz_obs - cz_pred)**2 / sigma_tot_sq)
        return lkl

    def grad_lkl(self, delta_k, A):
        lkl = grad(self.log_lkl, 0)(delta_k, A)
        return jnp.array([-lkl[0], lkl[1]])

    def cosmo_lnprob(self, A, delta_k):
        ln_prob = -self.log_lkl(delta_k, A) - 0.5 * ((A - 1.)/0.3)**2
        return ln_prob
