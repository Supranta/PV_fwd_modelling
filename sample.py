import sys
import numpy as np
import h5py as h5

from fwd_PV.chi_squared import ChiSquared
from fwd_PV.samplers.hmc import HMCSampler
from fwd_PV.tools.cosmo import camb_PS
from fwd_PV.io import process_datafile, process_config

restart_flag = sys.argv[1]
assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_GRID, L_BOX,\
        N_MCMC, dt, N_LEAPFROG,\
        datafile, savedir, N_SAVE, N_RESTART= process_config(configfile)

r_hMpc, e_rhMpc, RA, DEC, z_obs = process_datafile(datafile, 'h5')

kh, pk = camb_PS()

ChiSquaredBox = ChiSquared(N_GRID, L_BOX, kh, pk, r_hMpc, e_rhMpc, RA, DEC, z_obs, interpolate=False)

if(restart_flag=='INIT'):
    delta_k = 0.1 * ChiSquaredBox.generate_delta_k()
    N_START = 0
elif(restart_flag=='RESUME'):
    print("Restarting run....")
    f_restart = h5.File(savedir+'/restart.h5', 'r')
    N_START = f_restart['N_STEP'].value
    delta_k = f_restart['delta_k'][:]
    f_restart.close()

mass_matrix = np.array([2. * ChiSquaredBox.V / ChiSquaredBox.Pk_3d, 2. * ChiSquaredBox.V / ChiSquaredBox.Pk_3d])
sampler = HMCSampler(delta_k.shape, ChiSquaredBox.psi, ChiSquaredBox.grad_psi, mass_matrix, verbose=True)
accepted = 0

dt = dt

for i in range(N_START, N_START + N_MCMC):
    delta_k, ln_prob, acc = sampler.sample_one_step(delta_k, dt, N_LEAPFROG)
    print("ln_prob: %2.4f"%(ln_prob))
    if(acc):
        print("Accepted")
        accepted += 1

    acceptance_rate = accepted / (i - N_START + 1)
    print("Current acceptance rate: %2.3f"%(acceptance_rate))
    if(i%N_SAVE==0):
        print('=============')
        print('Saving file...')
        print('=============')
        j = i//N_SAVE
        f = h5.File(savedir + '/mcmc_'+str(j)+'.h5', 'w')
        f['delta_k'] = delta_k
        f['ln_prob'] = ln_prob
        f.close()
    if(i%N_RESTART==0):
        print("Saving restart file...")
        f = h5.File(savedir+'/restart.h5', 'w')
        f['delta_k'] = delta_k
        f['N_STEP'] = i
        f.close()
