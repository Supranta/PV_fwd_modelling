import numpy as np
import jax.numpy as jnp
from .tools.cosmo import z_cos, speed_of_light
from astropy.coordinates import SkyCoord
import astropy.units as u

from fwd_PV.velocity_box import ForwardModelledVelocityBox

class ChiSquared(ForwardModelledVelocityBox):
    def __init__(self, N_SIDE, L_BOX, kh, pk, r_hMpc, RA, DEC, z_obs, interpolate=False):
        super().__init__(N_SIDE, L_BOX, kh, pk)
        r_hat = np.array(SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).cartesian.xyz)
        self.r_hat = r_hat
        self.cz_obs = speed_of_light * z_obs
        self.cartesian_pos = r_hMpc * r_hat
        self.z_cos = z_cos(r_hMpc, self.OmegaM)
        self.indices = ((self.cartesian_pos +  self.L_BOX/2.) / self.l).astype(int)
        self.sig_v = 150.
        self.interpolate = interpolate
        if(self.interpolate):
            self.xyz_d = (self.cartesian_pos +  self.L_BOX/2.) / self.l - self.indices
        
    def log_lkl(self, delta_k):
        V_r = self.Vr_grid(delta_k)
        if(self.interpolate):
            V_r_tracers = self.V_interpolate(V_r)
        else:
            V_r_tracers = V_r[self.indices[0], self.indices[1], self.indices[2]]
        cz_pred = speed_of_light * self.z_cos + V_r_tracers * (1. + self.z_cos)
        return jnp.sum(0.5 * (self.cz_obs - cz_pred)**2 / self.sig_v / self.sig_v)
    
    def V_interpolate(self, V_r):
        c00 = (1. - self.xyz_d[0]) * V_r[self.indices[0], self.indices[1], self.indices[2]] + self.xyz_d[0] * V_r[self.indices[0]+1, self.indices[1], self.indices[2]]
        c01 = (1. - self.xyz_d[0]) * V_r[self.indices[0], self.indices[1], self.indices[2]+1] + self.xyz_d[0] * V_r[self.indices[0]+1, self.indices[1], self.indices[2]+1]
        c10 = (1. - self.xyz_d[0]) * V_r[self.indices[0], self.indices[1]+1, self.indices[2]] + self.xyz_d[0] * V_r[self.indices[0]+1, self.indices[1]+1, self.indices[2]]
        c11 = (1. - self.xyz_d[0]) * V_r[self.indices[0], self.indices[1]+1, self.indices[2]+1] + self.xyz_d[0] * V_r[self.indices[0]+1, self.indices[1]+1, self.indices[2]+1]
        
        c0 = (1. - self.xyz_d[1]) * c00 + self.xyz_d[1] * c01
        c1 = (1. - self.xyz_d[1]) * c10 + self.xyz_d[1] * c11
        
        return (1. - self.xyz_d[2]) * c0 + self.xyz_d[2] * c1