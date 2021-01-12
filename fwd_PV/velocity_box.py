import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks
from .tools.cosmo import camb_PS, get_Pk

class ForwardModelledVelocityBox:
    def __init__(self, N_GRID, L_BOX, smoothing_scale, window, Pk_type, Pk_data=None):
        self.L_BOX  = L_BOX
        self.N_GRID = N_GRID
        self.V = L_BOX**3
        l = L_BOX/N_GRID
        self.J = jnp.array(np.complex(0, 1))
        self.l = l
        self.dV = l**3
        self.kh, self.pk_camb = camb_PS()
        self.OmegaM_true = 0.315
        #self.sigma8_true = 0.7791
        self.sigma8_true = 0.8154
        self.k, self.k_norm = Fourier_ks(N_GRID, l)
        if(Pk_type=='simple'):
            self.sigma8_fid = self.sigma8_true
        elif(Pk_type=='camb_interpolate'):
            self.s8_arr, self.Om_arr, self.Pk_arr, self.kh = Pk_data 
        self.Pk_type = Pk_type
        if(window=='Gaussian'):
            kR = self.k_norm * smoothing_scale
            self.window = np.exp(-0.5 * kR**2)
            self.kmax = 0.1
        elif(window=='kmax'):
            window_kmax = np.ones(self.k_norm.shape)
            window_kmax[self.k_norm > smoothing_scale] = 0.
            self.window = window_kmax
            self.window_imag = window_kmax
            print("self.window_imag.shape: "+str(self.window_imag.shape))
            self.window_imag[:,:,-1] = 0.
            self.kmax = smoothing_scale
        self.r_hat_grid = grid_r_hat(N_GRID)
        self.Pk_interp = interp1d(self.kh, self.pk_camb)
        self.Pk_3d = self.get_Pk_3d()

    def get_Pk_3d(self):
        Pk_3d = 1e-20 * np.ones(self.k_norm.shape)
        select_positive_k = (self.k_norm > 1e-10)
        Pk_3d[select_positive_k] = self.Pk_interp(self.k_norm[select_positive_k])
        return Pk_3d

    def Pk_var(self, sigma8, OmegaM, get_pk=False):
        if(self.Pk_type=='simple'):
            pk = (sigma8 / self.sigma8_fid)**2 * self.pk_camb
        elif(self.Pk_type=='camb_interpolate'):
            pk = get_Pk(sigma8, OmegaM, self.s8_arr, self.Om_arr, self.Pk_arr)
        Pk_3d = 1e-20 * np.ones(self.k_norm.shape)
        select_positive_k = (self.k_norm > 1e-15)
        interp_pk = interp1d(self.kh, pk)
        Pk_3d[select_positive_k] = interp_pk(self.k_norm[select_positive_k])
        if(get_pk):
            return jnp.array(Pk_3d), pk
        return jnp.array(Pk_3d)

    def generate_delta_k(self):
        Pk_var_3d, pk = self.Pk_var(self.sigma8_true, self.OmegaM_true, get_pk=True) 
        delta_k_var = Pk_var_3d / self.V / 2.

        delta_k_real = np.random.normal(0., np.sqrt(delta_k_var))
        delta_k_imag = np.random.normal(0., np.sqrt(delta_k_var))
        
        print("delta_k_var.shape: "+str(delta_k_var.shape))
        delta_k_real[0,0,0] = 0.
        delta_k_imag[0,0,0] = 0. 
        delta_k_imag[:,:,-1] = 0.
        
        return jnp.array([delta_k_real, delta_k_imag])

    def get_delta_grid(self, delta_k):
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        delta_x = self.V / self.dV * jnp.fft.irfftn(delta_k_complex)
        return delta_x
   
    @partial(jit, static_argnums=(0,)) 
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
        delta_k_var = self.Pk_var(sigma8, OmegaM) / self.V / 2.
        ln_prior_real = jnp.sum(0.5 * self.window * (delta_k[0]**2) / delta_k_var) + 0.5 * jnp.sum(self.window * jnp.log(delta_k_var))
        ln_prior_imag = jnp.sum(0.5 * self.window_imag * (delta_k[1]**2) / delta_k_var) + 0.5 * jnp.sum(self.window_imag * jnp.log(delta_k_var))
        ln_prior = ln_prior_real + ln_prior_imag
        return ln_prior

    def grad_prior(self, delta_k, sigma8, OmegaM):
        return grad(self.log_prior, 0)(delta_k, sigma8, OmegaM)
    
    def psi(self, delta_k, sigma8, OmegaM, sig_v):
        return self.log_prior(delta_k, sigma8, OmegaM) + self.log_lkl(delta_k, OmegaM, sig_v)

    def grad_psi(self, delta_k, sigma8, OmegaM, sig_v):
        return self.grad_prior(delta_k, sigma8, OmegaM) + self.grad_lkl(delta_k, OmegaM, sig_v)

    def lnprob_s8(self, sigma8, OmegaM, delta_k):
        #print("sigma8: %2.7f, Omega_m: %2.7f"%(sigma8, OmegaM))
        if((sigma8 < 0.3)or(sigma8 > 1.2)):
            return -np.inf
        logP = -self.log_prior(delta_k, sigma8, OmegaM)
        #print("logP: %2.3f"%(logP))
        return logP
