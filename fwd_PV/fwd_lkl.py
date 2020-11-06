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
        self.r_hMpc = r_hMpc.reshape((1,-1))
        self.e_rhMpc = e_rhMpc.reshape((1,-1))
        self.z_cos = z_cos(self.r, self.OmegaM)
        self.cz_obs = (speed_of_light * z_obs).reshape((1,-1))
        self.sig_v = 250.
        

    def get_los_density(self, delta_MB, L_BOX_MB, N_GRID_MB, N_POINTS=201):
        pass
