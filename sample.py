import sys
import numpy as np
import h5py as h5

from fwd_PV.chi_squared import ChiSquared
from fwd_PV.fwd_lkl import ForwardLikelihoodBox
from fwd_PV.samplers.hmc import HMCSampler
from fwd_PV.samplers.slice import SliceSampler
from fwd_PV.tools.cosmo import camb_PS
from fwd_PV.io import process_datafile, process_config, config_fwd_lkl
from jax.config import config
config.update("jax_enable_x64", True)

restart_flag = sys.argv[1]
assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_GRID, L_BOX, likelihood, sample_cosmology,\
        N_MCMC, dt, N_LEAPFROG,\
        datafile, savedir, N_SAVE, N_RESTART= process_config(configfile)

assert likelihood == 'chi-squared' or likelihood == 'fwd_lkl', "The likelihood must be chi-squared or forward-likelihood."

r_hMpc, e_rhMpc, RA, DEC, z_obs = process_datafile(datafile, 'h5')

kh, pk = camb_PS()

if(likelihood=='chi-squared'):
    print("Initializing Chi-Squared Velocity Box....")
    VelocityBox = ChiSquared(N_GRID, L_BOX, kh, pk, r_hMpc, e_rhMpc, RA, DEC, z_obs, interpolate=False)
elif(likelihood=='fwd_lkl'):
    print("Initializing fwd_lkl Velocity Box....")
    PV_data = [r_hMpc, e_rhMpc, RA, DEC, z_obs]
    MB_data = config_fwd_lkl(configfile)
    VelocityBox = ForwardLikelihoodBox(N_GRID, L_BOX, kh, pk, PV_data, MB_data)
    
if(restart_flag=='INIT'):
    density_scaling = 0.1
    delta_k = density_scaling * VelocityBox.generate_delta_k()
    A = 1.
    N_START = 0

elif(restart_flag=='RESUME'):
    print("Restarting run....")
    f_restart = h5.File(savedir+'/restart.h5', 'r')
    N_START = f_restart['N_STEP'].value
    delta_k = f_restart['delta_k'][:]
    A = f_restart['A'].value
    f_restart.close()

mass_matrix = np.array([2. * VelocityBox.V / VelocityBox.Pk_3d, 2. * VelocityBox.V / VelocityBox.Pk_3d])
sampler = HMCSampler(delta_k.shape, VelocityBox.psi, VelocityBox.grad_psi, mass_matrix, verbose=True)
A_sampler = SliceSampler(1, VelocityBox.cosmo_lnprob, 0.1)
accepted = 0

N_THIN_COSMO = 2

dt = dt

for i in range(N_START, N_START + N_MCMC):
    delta_k, ln_prob, acc = sampler.sample_one_step(delta_k, dt, N_LEAPFROG, psi_kwargs={"A":A}, grad_psi_kwargs={"A":A})
    print("ln_prob: %2.4f"%(ln_prob))
    if(acc):
        print("Accepted")
        accepted += 1

    acceptance_rate = accepted / (i - N_START + 1)
    print("Current acceptance rate: %2.3f"%(acceptance_rate))
    if(sample_cosmology & (i%N_THIN_COSMO == 0)):
        print("Sampling cosmology...")
        A = A_sampler.sample_one_step(A, lnprob_kwargs={"delta_k":delta_k})
        print("A: %2.4f"%(A))
    if(i%N_SAVE==0):
        print('=============')
        print('Saving file...')
        print('=============')
        j = i//N_SAVE
        f = h5.File(savedir + '/mcmc_'+str(j)+'.h5', 'w')
        f['delta_k'] = delta_k
        f['ln_prob'] = ln_prob
        f['A'] = A
        f.close()
    if(i%N_RESTART==0):
        print("Saving restart file...")
        f = h5.File(savedir+'/restart.h5', 'w')
        f['delta_k'] = delta_k
        f['N_STEP'] = i
        f['A'] = A
        f.close()
