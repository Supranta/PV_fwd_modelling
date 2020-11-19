import sys
import numpy as np
import h5py as h5
import time

from fwd_PV.chi_squared import ChiSquared
from fwd_PV.fwd_lkl import ForwardLikelihoodBox
from fwd_PV.samplers.hmc import HMCSampler
from fwd_PV.samplers.slice import SliceSampler
from fwd_PV.tools.cosmo import camb_PS
from fwd_PV.io import process_datafile, process_config, config_fwd_lkl, config_Pk
from jax.config import config
config.update("jax_enable_x64", True)

restart_flag = sys.argv[1]
assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_GRID, L_BOX, likelihood, sample_cosmology, sample_sigv, window, smoothing_scale, Pk_type,\
        N_MCMC, dt, N_LEAPFROG,\
        datafile, savedir, N_SAVE, N_RESTART= process_config(configfile)

assert likelihood == 'chi-squared' or likelihood == 'fwd_lkl', "The likelihood must be chi-squared or forward-likelihood."

if(Pk_type=='simple'):
    Pk_data = None
elif(Pk_type=='camb_interpolate'):
    with h5.File(savedir+'/Pk_arr.h5', 'r') as f:
        Om_mesh = f['Om_mesh'][:]
        s8_mesh = f['s8_mesh'][:]
        i_mesh = f['i_mesh'][:]
        j_mesh = f['j_mesh'][:]
        Pk_arr = f['Pk_arr'][:] 
        factor = f['factor'].value
    Pk_data = [s8_mesh, Om_mesh, Pk_arr, i_mesh, j_mesh, factor]

r_hMpc, e_rhMpc, RA, DEC, z_obs = process_datafile(datafile, 'h5')

if(likelihood=='chi-squared'):
    print("Initializing Chi-Squared Velocity Box....")
    VelocityBox = ChiSquared(N_GRID, L_BOX, r_hMpc, e_rhMpc, RA, DEC, z_obs, smoothing_scale)
elif(likelihood=='fwd_lkl'):
    print("Initializing fwd_lkl Velocity Box....")
    PV_data = [r_hMpc, e_rhMpc, RA, DEC, z_obs]
    MB_data = config_fwd_lkl(configfile)
    VelocityBox = ForwardLikelihoodBox(N_GRID, L_BOX, PV_data, MB_data, smoothing_scale, window, Pk_type, Pk_data)
 
if(restart_flag=='INIT'):
    density_scaling = 0.1
    delta_k = density_scaling * VelocityBox.generate_delta_k()
    sigma8 = 0.8
    OmegaM = 0.315
    sig_v = 160.
    N_START = 0

elif(restart_flag=='RESUME'):
    print("Restarting run....")
    f_restart = h5.File(savedir+'/restart.h5', 'r')
    N_START = f_restart['N_STEP'].value
    delta_k = f_restart['delta_k'][:]
    try:
        sigma8 = f_restart['sigma8'].value
        OmegaM = f_restart['OmegaM'].value
        sig_v = f_restart['sig_v'].value
    except:
        sigma8 = 0.8
        OmegaM = 0.315
        sig_v = 160.
    f_restart.close()

mass_matrix = np.array([2. * VelocityBox.V / VelocityBox.Pk_3d, 2. * VelocityBox.V / VelocityBox.Pk_3d])
density_sampler = HMCSampler(delta_k.shape, VelocityBox.psi, VelocityBox.grad_psi, mass_matrix, verbose=True)
accepted = 0

if(sample_cosmology):
    s8_sampler = SliceSampler(1, VelocityBox.lnprob_s8, 0.03)
    Om_sampler = SliceSampler(1, VelocityBox.lnprob_Om, 0.03)
if(sample_sigv):
    sigv_sampler = SliceSampler(1, VelocityBox.lnprob_sigv, 10.)
dt = dt

for i in range(N_START, N_START + N_MCMC):
    start_time=time.time()
    delta_k, ln_prob, acc = density_sampler.sample_one_step(delta_k, dt, N_LEAPFROG, psi_kwargs={"sigma8":sigma8, "OmegaM": OmegaM, "sig_v":sig_v}, grad_psi_kwargs={"sigma8":sigma8, "OmegaM":OmegaM, "sig_v":sig_v})
    print("ln_prob: %2.4f"%(ln_prob))
    if(acc):
        print("Accepted")
        accepted += 1
    end_time = time.time()
    print("Time taken: %2.4f"%(end_time - start_time))
    acceptance_rate = accepted / (i - N_START + 1)
    print("Current acceptance rate: %2.3f"%(acceptance_rate))
    if(sample_cosmology):
        print("Sampling cosmology...")
        sigma8 = s8_sampler.sample_one_step(sigma8, lnprob_kwargs={"delta_k":delta_k, "OmegaM":OmegaM})
        OmegaM = Om_sampler.sample_one_step(OmegaM, lnprob_kwargs={"delta_k":delta_k, "sigma8":sigma8, "sig_v":sig_v})
        print("sigma8: %2.4f, OmegaM:%2.4f"%(sigma8, OmegaM))
    if(sample_sigv):
        print("Sample sig_v...")
        sig_v = sigv_sampler.sample_one_step(sig_v, lnprob_kwargs={"delta_k":delta_k, "OmegaM":OmegaM, "sigma8":sigma8})
        print("sig_v: %2.2f"%(sig_v))
    if(i%N_SAVE==0):
        print('=============')
        print('Saving file...')
        print('=============')
        j = i//N_SAVE
        f = h5.File(savedir + '/mcmc_'+str(j)+'.h5', 'w')
        f['delta_k'] = delta_k
        f['ln_prob'] = ln_prob
        f['sigma8'] = sigma8
        f['OmegaM'] = OmegaM
        f['sig_v'] = sig_v
        f.close()
    if(i%N_RESTART==0):
        print("Saving restart file...")
        f = h5.File(savedir+'/restart.h5', 'w')
        f['delta_k'] = delta_k
        f['N_STEP'] = i
        f['sigma8'] = sigma8
        f['OmegaM'] = OmegaM
        f['sig_v'] = sig_v
        f.close()
