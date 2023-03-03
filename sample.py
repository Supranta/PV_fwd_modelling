import sys
import jax.numpy as jnp
import numpy as np
import h5py as h5
import time

from fwd_PV.chi_squared import ChiSquared
from fwd_PV.fwd_lkl import ForwardLikelihoodBox
from fwd_PV.samplers import HMCSampler, SliceSampler
from fwd_PV.tools.cosmo import camb_PS
import fwd_PV.io as io
from jax.config import config
config.update("jax_enable_x64", True)

restart_flag = sys.argv[1]
assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_GRID, L_BOX, likelihood, coord_system_box = io.config_box(configfile)
N_MCMC, dt, N_LEAPFROG, sample_scale        = io.config_mcmc(configfile)
savedir, N_SAVE, N_RESTART                  = io.config_io(configfile)
N_CAT, PV_data, data_len                    = io.config_data(configfile)

assert likelihood == 'chi-squared' or likelihood == 'fwd_lkl', "The likelihood must be chi-squared or forward-likelihood."

kh, pk = camb_PS()

if(likelihood=='chi-squared'):
    print("Initializing Chi-Squared Velocity Box....")
    VelocityBox = ChiSquared(N_GRID, L_BOX, kh, pk, PV_data)
elif(likelihood=='fwd_lkl'):
    print("Initializing fwd_lkl Velocity Box....")    
    MB_data = io.config_fwd_lkl(configfile)
    VelocityBox = ForwardLikelihoodBox(N_GRID, L_BOX, kh, pk, PV_data, MB_data, coord_system_box)

VelocityBox.N_CAT = N_CAT
VelocityBox.data_len = jnp.array(data_len)

OmegaM = 0.315
sig_v = 150.    

if(restart_flag=='INIT'):
    density_scaling = 0.1
    delta_k = density_scaling * VelocityBox.generate_delta_k()    
    scale = jnp.ones(N_CAT)
    N_START = 0

elif(restart_flag=='RESUME'):
    print("Restarting run....")
    f_restart = h5.File(savedir+'/restart.h5', 'r')
    N_START = f_restart['N_STEP'][()]
    delta_k = f_restart['delta_k'][:]
    try:
        scale   = f_restart['scale'][:]
    except:
        scale = jnp.ones(N_CAT)
        
    f_restart.close()

mass_matrix = 2. * VelocityBox.V / VelocityBox.Pk_3d
#mass_matrix = np.load(savedir+'/mass_matrix.npy')
density_sampler = HMCSampler(delta_k.shape, VelocityBox.psi, VelocityBox.grad_psi, mass_matrix, verbose=True)
accepted = 0

if(sample_scale):
    scale_sampler = SliceSampler(N_CAT, VelocityBox.log_lkl_scale, verbose=True)
    scale_sampler.set_cov(np.diag(1e-4 * np.ones(N_CAT)))    
dt = dt

for i in range(N_START, N_START + N_MCMC):
    print("==================")
    print("MCMC step: %d"%(i))
    print("==================")
    start_time=time.time()
    delta_k, ln_prob, acc = density_sampler.sample_one_step(delta_k, dt, N_LEAPFROG, psi_kwargs={"scale": scale}, grad_psi_kwargs={"scale": scale})
    print("ln_prob: %2.4f"%(ln_prob))    
    if(acc):
        print("Accepted")
        accepted += 1    
    end_time = time.time()
    print("Time taken: %2.4f seconds"%(end_time - start_time))
    acceptance_rate = accepted / (i - N_START + 1)
    print("Current acceptance rate: %2.3f"%(acceptance_rate))
    if(i%N_SAVE==0):
        io.write_save_file(i, N_SAVE, savedir, delta_k, ln_prob, scale)
    if(sample_scale):
        print("Sampling scale...")
        scale, _, _ = scale_sampler.sample_one_step(scale, lnprob_kwargs={"delta_k": delta_k})
        print("scale: " + str(scale))                
    if(i%N_RESTART==0):
        io.write_restart_file(savedir, delta_k, i, scale)
        
        
"""
NOTES

Data from
https://arizona.app.box.com/s/w6qs5wknnqtznaexc33yvyvybmhh6kco

Can make Gaussian mocks using
https://github.com/Supranta/Pantheon-plus/blob/master/LinearMocks/make_gaussian_mock.py
but may be in different format
"""
