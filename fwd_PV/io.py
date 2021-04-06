import pandas as pd
import numpy as np
import h5py as h5
import configparser

def config_box(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    N_GRID = int(config['BOX']['N_GRID'])
    L_BOX  = float(config['BOX']['L_BOX'])
    likelihood = config['BOX']['likelihood']
    
    return N_GRID, L_BOX, likelihood

def config_mcmc(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
   
    N_MCMC = int(config['MCMC']['N_MCMC'])
    dt     = float(config['MCMC']['dt'])
    N_LEAPFROG = int(config['MCMC']['N_LEAPFROG'])
    
    return N_MCMC, dt, N_LEAPFROG

def config_io(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    datafile = config['IO']['datafile']    
    savedir  = config['IO']['savedir']    
    N_SAVE = int(config['IO']['N_SAVE'])
    N_RESTART = int(config['IO']['N_RESTART'])

    return datafile, savedir, N_SAVE, N_RESTART 


def process_datafile(datafile, filetype='csv'):
    if(filetype=='csv'):
        df = pd.read_csv(datafile)

        r_hMpc = np.array(df['r_hMpc'])
        RA = np.array(df['RA'])
        DEC = np.array(df['DEC'])
        z_obs = np.array(df['z_obs'])
    elif(filetype=='h5'):
        with h5.File(datafile, 'r') as f:
            r_hMpc = f['r_hMpc'][:]
            e_r_hMpc = f['e_rhMpc'][:]
            RA     = f['RA'][:]
            DEC    = f['DEC'][:]
            z_obs  = f['z_obs'][:]

    return r_hMpc, e_r_hMpc, RA, DEC, z_obs

def config_fwd_lkl(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    N_GRID = int(config['FWD_LKL']['N_BOX_sim'])
    L_BOX  = float(config['FWD_LKL']['L_BOX_sim'])
    
    density_data = config['FWD_LKL']['density_file']
    print("Loading density data from "+density_data)
    delta_grid = np.load(density_data)
    
    return delta_grid, L_BOX, N_GRID
          
def write_save_file(i, N_SAVE, savedir, delta_k, ln_prob):
    print('=============')
    print('Saving file...')
    print('=============')
    j = i//N_SAVE
    with h5.File(savedir + '/mcmc_'+str(j)+'.h5', 'w') as f:
        f['delta_k'] = delta_k
        f['ln_prob'] = ln_prob
        
def write_restart_file(savedir, delta_k, i):
    print("Saving restart file...")
    with h5.File(savedir+'/restart.h5', 'w') as f:
        f['delta_k'] = delta_k
        f['N_STEP'] = i