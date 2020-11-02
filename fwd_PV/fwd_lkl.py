import numpy as np
import jax.numpy as jnp
from jax import grad
from .tools.cosmo import z_cos, speed_of_light
from astropy.coordinates import SkyCoord
import astropy.units as u

from fwd_PV.velocity_box import ForwardModelledVelocityBox

EPS = 1e-50

class ForwardLikelihoodBox(ForwardModelledVelocityBox):
    def __init__(self, N_SIDE, L_BOX, kh, pk, PV_data, MB_data, N_POINTS=201):
        super().__init__(N_SIDE, L_BOX, kh, pk)
        r_hMpc, e_rhMpc, RA, DEC, z_obs = PV_data
        r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).cartesian.xyz)
        self.r_hat = r_hat
        self.sigmad = e_rhMpc * 100.
        self.RA  = RA
        self.DEC = DEC
        delta_MB, L_BOX_MB, N_GRID_MB = MB_data
        self.los_density = self.get_los_density(delta_MB, L_BOX_MB, N_GRID_MB, N_POINTS)
        r = np.linspace(1., 200., N_POINTS)
        self.r = r.reshape((N_POINTS, 1))
        self.delta_r = np.mean((r[1:] - r[:-1]))
        self.r_hMpc = r_hMpc.reshape((1,-1))
        self.e_rhMpc = e_rhMpc.reshape((1,-1))
        self.z_cos = z_cos(self.r, self.OmegaM)
        self.cz_obs = (speed_of_light * z_obs).reshape((1,-1))
        self.sig_v = 250.
        

    def get_los_density(self, delta_MB, L_BOX_MB, N_GRID_MB, N_POINTS=201):
        r_hat = np.array(SkyCoord(ra=self.RA*u.deg, dec=self.DEC*u.deg).cartesian.xyz)
        r_hat = r_hat.reshape((1,3,-1))
        r = np.linspace(1., 200., N_POINTS)
        r = r.reshape((N_POINTS, 1, 1))
        cartesian_pos = (r * r_hat)
        l  = L_BOX_MB / N_GRID_MB
        MB_indices = ((cartesian_pos + L_BOX_MB / 2.) / l).astype(int)
        self.indices = ((cartesian_pos + self.L_BOX / 2.) / self.l).astype(int)
        delta_los = delta_MB[MB_indices[:,0,:], MB_indices[:,1,:], MB_indices[:,2,:]]
        return delta_los
    
    def log_lkl(self, delta_k):
        V_r = self.Vr_grid(delta_k)
        Vr_los = V_r[self.indices[:,0,:], self.indices[:,1,:], self.indices[:,2,:]]
        cz_pred = speed_of_light * self.z_cos + (1. + self.z_cos) * Vr_los
        delta_cz = (cz_pred - self.cz_obs)
        p_r = np.exp(-0.5 * ((self.r - self.r_hMpc)/self.e_rhMpc)**2) * (1. + self.los_density)
        if(np.sum(np.isnan(p_r)) > 0):
            for i in range(20):
                print("NAN DETECTED")
        return jnp.sum(jnp.log(EPS + jnp.trapz(self.delta_r * p_r * jnp.exp(-0.5 * delta_cz**2 / self.sig_v**2), self.r, axis=0)))

    def grad_lkl(self, delta_k):
        lkl = grad(self.log_lkl)(delta_k)
        return jnp.array([-lkl[0], lkl[1]])