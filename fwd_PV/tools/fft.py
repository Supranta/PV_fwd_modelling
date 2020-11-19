import numpy as np
import jax.numpy as jnp

def Fourier_ks(N_BOX, l):
    kx = 2*np.pi*np.fft.fftfreq(N_BOX,d=l)
    ky = 2*np.pi*np.fft.fftfreq(N_BOX,d=l)
    kz = 2*np.pi*np.fft.fftfreq(N_BOX,d=l)
    
    N_BOX_Z = (N_BOX//2 +1)
    
    kx_vec = np.tile(kx[:, None, None], (1, N_BOX, N_BOX_Z))
    ky_vec = np.tile(ky[None, :, None], (N_BOX, 1, N_BOX_Z))
    kz_vec = np.tile(kz[None, None, :N_BOX_Z], (N_BOX, N_BOX, 1))

    k_norm = np.sqrt(kx_vec**2 + ky_vec**2 + kz_vec**2)
    k_norm[(k_norm < 1e-10)] = 1e-15
    
    return np.array([kx_vec, ky_vec, kz_vec]), k_norm

def grid_r_hat(N_BOX):
    X = np.linspace(-N_BOX/2, N_BOX/2, N_BOX)
    Y = np.linspace(-N_BOX/2, N_BOX/2, N_BOX)
    Z = np.linspace(-N_BOX/2, N_BOX/2, N_BOX)

    X_3d = np.tile(X[:, None, None], (1, N_BOX, N_BOX))
    Y_3d = np.tile(Y[None, :, None], (N_BOX, 1, N_BOX))
    Z_3d = np.tile(Z[None, None, :], (N_BOX, N_BOX, 1))

    R_vec = np.array([X_3d, Y_3d, Z_3d])

    R_hat = R_vec / np.linalg.norm(R_vec, axis=0)
    
    return jnp.array(R_hat)
