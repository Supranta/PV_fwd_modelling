import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks

class ForwardModelledVelocityBox:
    def __init__(self, N_SIDE, L_BOX, kh, pk):
        self.L_BOX  = L_BOX
        self.N_SIDE = N_SIDE
        self.V = L_BOX**3
        l = L_BOX/N_SIDE
        self.J = jnp.array(np.complex(0, 1))
        self.l = l
        self.dV = l**3
        self.k, self.k_norm = Fourier_ks(N_SIDE, l)
        smoothing_scale = 4.
        self.sig_v      = 150.
        kR = self.k_norm * smoothing_scale
        self.window = np.exp(-0.5 * kR**2)        
        OmegaM = 0.315
        self.OmegaM = OmegaM
        self.f = OmegaM**0.55
        self.r_hat_grid = grid_r_hat(N_SIDE)
        self.Pk_interp = interp1d(kh, pk)
        self.Pk_3d = self.get_Pk_3d()

    def get_Pk_3d(self):
        k_abs = np.array(self.k_norm)
        Pk_3d = 1e-20 * np.ones(k_abs.shape)
        select_positive_k = (k_abs > 1e-10)
        Pk_3d[select_positive_k] = self.Pk_interp(k_abs[select_positive_k])
        return jnp.array(Pk_3d)

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
    
    def Vr_grid(self, delta_k):
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        
        f = self.f

        v_kx = self.J * 100 * self.window * f * delta_k_complex * self.k[0] / self.k_norm / self.k_norm
        v_ky = self.J * 100 * self.window * f * delta_k_complex * self.k[1] / self.k_norm / self.k_norm
        v_kz = self.J * 100 * self.window * f * delta_k_complex * self.k[2] / self.k_norm / self.k_norm

        vx = (jnp.fft.irfftn(v_kx) * self.V / self.dV)
        vy = (jnp.fft.irfftn(v_ky) * self.V / self.dV)
        vz = (jnp.fft.irfftn(v_kz) * self.V / self.dV)

        V = jnp.array([vx, vy, vz])

        return jnp.sum(V * self.r_hat_grid, axis=0)

    def log_prior(self, delta_k):
        delta_k_var = self.Pk_3d / self.V / 2.
        ln_prior = jnp.sum(0.5 * (delta_k[0]**2 + delta_k[1]**2) / delta_k_var) + jnp.sum(jnp.log(delta_k_var))
        return ln_prior

    def grad_prior(self, delta_k):
        return grad(self.log_prior, 0)(delta_k)

    def psi(self, delta_k):
        return self.log_prior(delta_k) + self.log_lkl(delta_k)

    def grad_psi(self, delta_k):
        return self.grad_prior(delta_k) + self.grad_lkl(delta_k)