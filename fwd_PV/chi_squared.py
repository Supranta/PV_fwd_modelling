import numpy as np
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
        
    def log_lkl(self, delta_k):
        V_r = self.Vr_grid(delta_k)
        V_r_tracers = V_r[self.indices[0], self.indices[1], self.indices[2]]
        cz_pred = speed_of_light * self.z_cos + V_r_tracers * (1. + self.z_cos)
        return np.sum(0.5 * (self.cz_obs - cz_pred)**2 / self.sig_v / self.sig_v)
    
    def grad_lkl(self, delta_k):
        V_r = self.Vr_grid(delta_k)
        V_r_tracers = V_r[self.indices[0], self.indices[1], self.indices[2]]
        cz_pred = speed_of_light * self.z_cos + V_r_tracers * (1. + self.z_cos)
        
        delta_cz = (self.cz_obs - cz_pred)
        
        A_x = np.zeros(V_r.shape)
        A_y = np.zeros(V_r.shape)
        A_z = np.zeros(V_r.shape)
        
        A = delta_cz / self.sig_v / self.sig_v * (1. + self.z_cos)
        
        A_x[self.indices[0], self.indices[1], self.indices[2]] += A * self.r_hat[0]
        A_y[self.indices[0], self.indices[1], self.indices[2]] += A * self.r_hat[1]
        A_z[self.indices[0], self.indices[1], self.indices[2]] += A * self.r_hat[2]
        
        B = self.J * 100. * self.f / self.V / self.k_norm / self.k_norm

        A_k_x = 2. * B * self.k[0] * np.fft.rfftn(A_x)
        A_k_y = 2. * B * self.k[1] * np.fft.rfftn(A_y)
        A_k_z = 2. * B * self.k[2] * np.fft.rfftn(A_z)
        
        grad = A_k_x + A_k_y + A_k_z

        return np.array([grad.real, grad.imag])
         
        