import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks, get_Fourier_mask, get_zero_imag_mask

class ForwardModelledVelocityBox:
    def __init__(self, N_SIDE, L_BOX, kh, pk):        
        self.set_up_box(L_BOX, N_SIDE)
        self.set_up_fft()
        self.set_up_cosmology(kh, pk)
        
        
        self.J = jnp.array(np.complex(0, 1))        
        
        smoothing_scale = 4.
        self.sig_v      = 150.
        kR = self.k_norm * smoothing_scale
        self.window = np.exp(-0.5 * kR**2)                
        self.r_hat_grid = grid_r_hat(N_SIDE)        
    
    def get_Pk_3d(self):
        k_abs = np.array(self.k_norm)
        Pk_3d = 1e-20 * np.ones(k_abs.shape)
        select_positive_k = (k_abs > 1e-10)
        Pk_3d[select_positive_k] = self.Pk_interp(k_abs[select_positive_k])
        Pk_real = Pk_3d
        Pk_imag = Pk_3d      
        return jnp.array([Pk_real, Pk_imag])

    def symmetrize(self, delta_k):
        delta_k = self.Fourier_mask[jnp.newaxis] * delta_k
        delta_k = jax.ops.index_update(delta_k, self.update_index_real_0,  jnp.take(jnp.flip(delta_k[0, 1:(self.N_Z-1), :, 0], axis=0), self.flip_indices, axis=1))
        delta_k = jax.ops.index_update(delta_k, self.update_index_real_ny,  jnp.take(jnp.flip(delta_k[0, 1:(self.N_Z-1), :, self.N_Z - 1], axis=0), self.flip_indices, axis=1))
        delta_k = jax.ops.index_update(delta_k, self.update_index_imag_0,  -jnp.take(jnp.flip(delta_k[1, 1:(self.N_Z-1), :, 0], axis=0), self.flip_indices, axis=1))
        delta_k = jax.ops.index_update(delta_k, self.update_index_imag_ny,  -jnp.take(jnp.flip(delta_k[1, 1:(self.N_Z-1), :, self.N_Z - 1], axis=0), self.flip_indices, axis=1))
        return delta_k
        
    def generate_delta_k(self):
        white_noise = np.random.normal(size=(self.N_SIDE,
                                             self.N_SIDE,
                                             self.N_SIDE))
        F_k = np.fft.rfftn(white_noise)
        delta_k = 1. / np.sqrt(self.N_SIDE**3)* F_k

        return np.sqrt(self.Pk_3d / self.V) * np.array([delta_k.real, delta_k.imag])
        
    def get_delta_grid(self, delta_k):
        delta_k = self.symmetrize(delta_k)
        delta_k_complex = delta_k[0] + self.J * delta_k[1]        
        delta_x = self.V / self.dV * jnp.fft.irfftn(delta_k_complex)
        return delta_x
    
    def Vr_grid(self, delta_k):
        delta_k = self.symmetrize(delta_k)
        
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        
        f = self.f

        v_kx = self.J * 100 * self.window * f * delta_k_complex * self.k[0] / self.k_norm / self.k_norm
        v_ky = self.J * 100 * self.window * f * delta_k_complex * self.k[1] / self.k_norm / self.k_norm
        v_kz = self.J * 100 * self.window * f * delta_k_complex * self.k[2] / self.k_norm / self.k_norm

        vx = (jnp.fft.irfftn(v_kx) * self.V / self.dV)
        vy = (jnp.fft.irfftn(v_ky) * self.V / self.dV)
        vz = (jnp.fft.irfftn(v_kz) * self.V / self.dV)

        velocity = jnp.array([vx, vy, vz])

        return jnp.sum(velocity * self.r_hat_grid, axis=0)

    def get_Pk(self, delta_k, k_bins):
        Pk_list = []
        k_bin_centre_list = []
        for i in range(len(k_bins) - 1):
            select_k = (self.k_norm > k_bins[i-1]) & (self.k_norm < k_bins[i]) & (self.Fourier_mask).astype(bool)
            k_bin_centre = np.mean(self.k_norm[select_k])
            k_bin_centre_list.append(k_bin_centre)
            Pk = np.mean(delta_k[0, select_k]**2 + delta_k[1, select_k]**2) * self.V
            Pk_list.append(Pk)
        return np.array(k_bin_centre_list), np.array(Pk_list)

    def log_prior(self, delta_k):
        delta_k_var = self.Pk_3d / self.V / 2.
        ln_prior = jnp.sum(0.5 * self.prior_mask * delta_k**2 / delta_k_var)
        return ln_prior

    def grad_prior(self, delta_k):
        return grad(self.log_prior, 0)(delta_k)
    
    def psi(self, delta_k, scale):
        return self.log_prior(delta_k) + self.log_lkl(delta_k, scale)

    def grad_psi(self, delta_k, scale):
        grad_arr = self.grad_prior(delta_k) + self.grad_lkl(delta_k, scale)
        grad_arr = np.array(grad_arr)
        grad_arr[:,0,0,0] = 0.
        return grad_arr
    
    def set_up_box(self, L_BOX, N_SIDE):
        self.L_BOX  = L_BOX
        self.N_SIDE = N_SIDE
        self.N_Z = N_SIDE//2 + 1
        self.V = L_BOX**3
        l = L_BOX/N_SIDE
        self.l = l
        self.dV = l**3
        
    def set_up_cosmology(self, kh, pk):
        OmegaM = 0.315
        self.OmegaM = OmegaM
        self.f = OmegaM**0.55
        self.Pk_interp = interp1d(kh, pk)
        self.Pk_3d = self.get_Pk_3d()
    
    def set_up_fft(self):
        self.k, self.k_norm = Fourier_ks(self.N_SIDE, self.l)
        mask = np.ones((self.N_SIDE,self.N_SIDE,self.N_Z))
        mask[self.N_Z:,:,0] = 0.
        mask[self.N_Z:,:,-1] = 0.
        self.Fourier_mask = mask
        self.prior_mask = np.array([mask, mask])
        zero_ny_ind = [0, self.N_Z - 1]
        for i in zero_ny_ind:
            for j in zero_ny_ind:
                for k in zero_ny_ind:
                    self.prior_mask[0,i,j,k] = 0.5 * self.prior_mask[0,i,j,k]
                    self.prior_mask[1,i,j,k] = 0.
        self.update_index_real_0  = jax.ops.index[0, self.N_Z:, :, 0]
        self.update_index_real_ny = jax.ops.index[0, self.N_Z:, :, self.N_Z - 1]
        self.update_index_imag_0  = jax.ops.index[1, self.N_Z:, :, 0]
        self.update_index_imag_ny = jax.ops.index[1, self.N_Z:, :, self.N_Z - 1]
        
        flip_indices = -np.arange(self.N_SIDE)
        flip_indices[self.N_Z - 1] = -flip_indices[self.N_Z - 1]
        self.flip_indices = flip_indices.tolist()