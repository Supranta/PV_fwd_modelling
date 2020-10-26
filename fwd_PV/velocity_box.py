import numpy as np
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks

class ForwardModelledVelocityBox:
    def __init__(self, N_SIDE, L_BOX, kh, pk):
        self.L_BOX  = L_BOX
        self.N_SIDE = N_SIDE
        self.V = L_BOX**3
        l = L_BOX/N_SIDE
        self.J = np.complex(0, 1)
        self.l = l
        self.dV = l**3
        self.k, self.k_norm = Fourier_ks(N_SIDE, l)
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
        return Pk_3d

    def generate_delta_k(self):
        delta_k = np.random.normal(0., np.sqrt(self.Pk_3d / self.V))
        delta_k[0,0,0] = 0.
        phi = np.random.uniform(0., 1., size=delta_k.shape)
        
        delta_k_real = delta_k * np.cos(phi)
        delta_k_imag = delta_k * np.sin(phi)
        
        return np.array([delta_k_real, delta_k_imag])

    def Vr_grid(self, delta_k):
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        
        v_kx = self.J * 100 * self.f * delta_k_complex * self.k[0] / self.k_norm / self.k_norm
        v_ky = self.J * 100 * self.f * delta_k_complex * self.k[1] / self.k_norm / self.k_norm
        v_kz = self.J * 100 * self.f * delta_k_complex * self.k[2] / self.k_norm / self.k_norm

        vx = (np.fft.irfftn(v_kx) * self.V / self.dV)
        vy = (np.fft.irfftn(v_ky) * self.V / self.dV)
        vz = (np.fft.irfftn(v_kz) * self.V / self.dV)

        V = np.array([vx, vy, vz])

        return np.sum(V * self.r_hat_grid, axis=0)

    def log_prior(self, delta_k):
        delta_k_var = self.Pk_3d / self.V
        ln_prior = np.sum(0.5 * (delta_k[0]**2 + delta_k[1]**2) / delta_k_var)
        print("log-prior per mode: %2.5f"%(2*ln_prior / len(delta_k.flatten())))
        return ln_prior
    
    def grad_prior(self, delta_k):
        delta_k_var = self.Pk_3d / self.V
        grad_real = delta_k[0]  / delta_k_var
        grad_imag = delta_k[1]  / delta_k_var
        return np.array([grad_real, grad_imag])

    def lkl(self, delta_k):
        pass
    
    def grad_lkl(self, delta_k):
        pass

    def psi(self, delta_k):
        return self.log_lkl(delta_k) + self.log_prior(delta_k)

    def grad_psi(self, delta_k):
        return self.grad_prior(delta_k) + self.grad_lkl(delta_k)
