import numpy as np
import h5py as h5

from fwd_PV.chi_squared import ChiSquared
from fwd_PV.samplers.hmc import HMCSampler
from fwd_PV.tools.cosmo import camb_PS
from fwd_PV.io import process_datafile

N_BOX = 64
L_BOX = 500.

N_MCMC = 2000

dt = 0.05
N = 10

datafile = 'data/VELMASS_mocks/mock_unique.csv'
savedir = 'fwd_PV_runs/sample_prior'
N_SAVE = 5

r_hMpc, RA, DEC, z_obs = process_datafile(datafile, 'h5')

kh, pk = camb_PS()

ChiSquaredBox = ChiSquared(N_BOX, L_BOX, kh, pk, r_hMpc, RA, DEC, z_obs, interpolate=False)
delta_k = 0.1 * ChiSquaredBox.generate_delta_k()

mass_matrix = np.array([2. * ChiSquaredBox.V / ChiSquaredBox.Pk_3d, 2. * ChiSquaredBox.V / ChiSquaredBox.Pk_3d])
sampler = HMCSampler(delta_k.shape, ChiSquaredBox.log_prior, ChiSquaredBox.grad_prior, mass_matrix, verbose=True)
accepted = 0

dt = dt

for i in range(N_MCMC):
    delta_k, ln_prob, acc = sampler.sample_one_step(delta_k, dt, N)
    print("ln_prob: %2.4f"%(ln_prob))
    if(acc):
        print("Accepted")
        accepted += 1

    acceptance_rate = accepted / (i + 1)
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
