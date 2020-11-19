import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks
from .tools.cosmo import Pk_interpolated, camb_PS

class ForwardModelledVelocityBox:
    def __init__(self, N_SIDE, L_BOX, smoothing_scale, window, Pk_type, Pk_data=None):
        self.L_BOX  = L_BOX
        self.N_SIDE = N_SIDE
        self.V = L_BOX**3
        l = L_BOX/N_SIDE
        self.J = jnp.array(np.complex(0, 1))
        self.l = l
        self.dV = l**3
        self.kh, pk = camb_PS()
        self.k, self.k_norm = Fourier_ks(N_SIDE, l)
        if(Pk_type=='simple'):
            self.sigma8_fid = 0.80
        elif(Pk_type=='camb_interpolate'):
            self.s8_mesh, self.Om_mesh, self.Pk_arr, self.i_mesh, self.j_mesh, self.factor = Pk_data 
        self.Pk_type = Pk_type
        if(window=='Gaussian'):
            kR = self.k_norm * smoothing_scale
            self.window = np.exp(-0.5 * kR**2)
        elif(window=='kmax'):
            window_kmax = np.ones(self.k_norm.shape)
            window_kmax[self.k_norm > smoothing_scale] = 0.
            self.window = window_kmax
        OmegaM = 0.315
        self.OmegaM = OmegaM
        self.f = OmegaM**0.55
        self.r_hat_grid = grid_r_hat(N_SIDE)
        self.Pk_interp = interp1d(self.kh, pk)
        self.Pk_3d = self.get_Pk_3d()

    def get_Pk_3d(self):
        Pk_3d = 1e-20 * np.ones(self.k_norm.shape)
        select_positive_k = (self.k_norm > 1e-10)
        Pk_3d[select_positive_k] = self.Pk_interp(self.k_norm[select_positive_k])
        return Pk_3d

    def Pk_var(self, sigma8, OmegaM):
        pk = Pk_interpolated(sigma8, OmegaM, self.s8_mesh, self.Om_mesh, self.Pk_arr, self.i_mesh, self.j_mesh, self.factor)
        Pk_3d = 1e-20 * np.ones(self.k_norm.shape)
        select_positive_k = (self.k_norm > 1e-10)
        interp_pk = interp1d(self.kh, pk)
        Pk_3d[select_positive_k] = interp_pk(self.k_norm[select_positive_k])
        return Pk_3d

    def generate_delta_k(self):
        delta_k_real = np.random.normal(0., np.sqrt(self.Pk_3d / self.V / 2))
        delta_k_imag = np.random.normal(0., np.sqrt(self.Pk_3d / self.V / 2))
        
        delta_k_real[0,0,0] = 0.
        delta_k_imag[0,0,0] = 0. 
        
        return jnp.array([delta_k_real, delta_k_imag])

    def get_delta_grid(self, delta_k):
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        delta_x = self.V / self.dV * jnp.fft.irfftn(delta_k_complex)
        return delta_x
    
    def Vr_grid(self, delta_k, OmegaM=0.315):
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        
        f = OmegaM**0.55

        v_kx = self.J * 100 * self.window * f * delta_k_complex * self.k[0] / self.k_norm / self.k_norm
        v_ky = self.J * 100 * self.window * f * delta_k_complex * self.k[1] / self.k_norm / self.k_norm
        v_kz = self.J * 100 * self.window * f * delta_k_complex * self.k[2] / self.k_norm / self.k_norm

        vx = (jnp.fft.irfftn(v_kx) * self.V / self.dV)
        vy = (jnp.fft.irfftn(v_ky) * self.V / self.dV)
        vz = (jnp.fft.irfftn(v_kz) * self.V / self.dV)

        V = jnp.array([vx, vy, vz])

        return jnp.sum(V * self.r_hat_grid, axis=0)

    def log_prior(self, delta_k, sigma8, OmegaM):
        if(self.Pk_type=='simple'):
            A = (sigma8 / self.sigma8_fid)**2
            delta_k_var = A * self.Pk_3d / self.V / 2.
        elif(self.Pk_type=='camb_interpolate'):
            delta_k_var = self.Pk_var(sigma8, OmegaM)
        ln_prior = jnp.sum(0.5 * (delta_k[0]**2 + delta_k[1]**2) / delta_k_var) + jnp.sum(jnp.log(delta_k_var))
        return ln_prior

    def grad_prior(self, delta_k, sigma8, OmegaM):
        return grad(self.log_prior, 0)(delta_k, sigma8, OmegaM)

    def psi(self, delta_k, sigma8, OmegaM, sig_v):
        return self.log_prior(delta_k, sigma8, OmegaM) + self.log_lkl(delta_k, OmegaM, sig_v)

    def grad_psi(self, delta_k, sigma8, OmegaM, sig_v):
        return self.grad_prior(delta_k, sigma8, OmegaM) + self.grad_lkl(delta_k, OmegaM, sig_v)

    def lnprob_s8(self, sigma8, OmegaM, delta_k):
        if((sigma8 < 0.3)or(sigma8 > 1.2)):
            return -np.inf
        select_low_k = (self.k_norm < 0.1)
        if(self.Pk_type=='simple'):
            A = (sigma8 / self.sigma8_fid)**2
            delta_k_var = A * self.Pk_3d / self.V / 2.
        elif(self.Pk_type=='camb_interpolate'):
            delta_k_var = self.Pk_var(sigma8, OmegaM)
        logP = -np.sum((0.5 * (delta_k[0]**2 + delta_k[1]**2) / delta_k_var)[select_low_k]) - np.sum(np.log(delta_k_var[select_low_k]))
        return logP
