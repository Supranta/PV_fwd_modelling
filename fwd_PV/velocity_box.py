import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks
from .tools.cosmo import camb_PS, get_Pk

J = jnp.array(np.complex(0,1))
sig_v = 230.

class ForwardModelledVelocityBox:
    def __init__(self, N_GRID, L_BOX):
        self.L_BOX  = L_BOX
        self.N_GRID = N_GRID
        self.V = L_BOX**3
        l = L_BOX/N_GRID
        self.l = l
        self.dV = l**3
        self.kh, self.pk_camb = camb_PS()
        self.OmegaM_true = 0.315
        self.sigma8_true = 0.8154
        self.k, self.k_norm = Fourier_ks(N_GRID, l)
        smoothing_scale = 4.
        kR = self.k_norm * smoothing_scale
        self.window = np.exp(-0.5 * kR**2)
        self.r_hat_grid = grid_r_hat(N_GRID)
        self.Pk_interp = interp1d(self.kh, self.pk_camb)
        self.Pk_3d = self.get_Pk_3d()

    def get_Pk_3d(self):
        Pk_3d = 1e-20 * np.ones(self.k_norm.shape)
        select_positive_k = (self.k_norm > 1e-10)
        Pk_3d[select_positive_k] = self.Pk_interp(self.k_norm[select_positive_k])
        return Pk_3d

    def generate_delta_k(self):
        Pk_var_3d = self.get_Pk_3d() 
        delta_k_var = Pk_var_3d / self.V / 2.

        delta_k_real = np.random.normal(0., np.sqrt(delta_k_var))
        delta_k_imag = np.random.normal(0., np.sqrt(delta_k_var))
        
        delta_k_real[0,0,0] = 0.
        delta_k_imag[0,0,0] = 0. 
        
        return jnp.array([delta_k_real, delta_k_imag])

    def get_delta_grid(self, delta_k):
        delta_k_complex = delta_k[0] + J * delta_k[1]
        delta_x = self.V / self.dV * jnp.fft.irfftn(delta_k_complex)
        return delta_x
   
    @partial(jit, static_argnums=(0,)) 
    def Vr_grid(self, delta_k, OmegaM=0.315):
        delta_k_complex = delta_k[0] + J * delta_k[1]
        
        f = OmegaM**0.55

        v_kx = J * 100 * self.window * f * delta_k_complex * self.k[0] / self.k_norm / self.k_norm
        v_ky = J * 100 * self.window * f * delta_k_complex * self.k[1] / self.k_norm / self.k_norm
        v_kz = J * 100 * self.window * f * delta_k_complex * self.k[2] / self.k_norm / self.k_norm

        vx = (jnp.fft.irfftn(v_kx) * self.V / self.dV)
        vy = (jnp.fft.irfftn(v_ky) * self.V / self.dV)
        vz = (jnp.fft.irfftn(v_kz) * self.V / self.dV)

        V = jnp.array([vx, vy, vz])

        return jnp.sum(V * self.r_hat_grid, axis=0)
    
    def log_prior(self, delta_k):
        delta_k_var = self.Pk_var() / self.V / 2.
        ln_prior_real = jnp.sum(0.5 * self.window * (delta_k[0]**2) / delta_k_var) + 0.5 * jnp.sum(self.window * jnp.log(delta_k_var))
        ln_prior_imag = jnp.sum(0.5 * self.window * (delta_k[1]**2) / delta_k_var) + 0.5 * jnp.sum(self.window * jnp.log(delta_k_var))
        ln_prior = ln_prior_real + ln_prior_imag
        return ln_prior

    def grad_prior(self, delta_k):
        return grad(self.log_prior, 0)(delta_k)
    
    def psi(self, delta_k, sigma8, OmegaM, sig_v):
        return self.log_prior(delta_k) + self.log_lkl(delta_k, sig_v)

    def grad_psi(self, delta_k, sig_v):
        return self.grad_prior(delta_k) + self.grad_lkl(delta_k, sig_v)
