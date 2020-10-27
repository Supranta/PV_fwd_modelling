import pandas as pd
import numpy as np
import h5py as h5
import configparser

def process_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    N_GRID = int(config['BOX']['N_GRID'])
    L_BOX  = float(config['BOX']['L_BOX'])

    N_MCMC = int(config['MCMC']['N_MCMC'])
    dt     = float(config['MCMC']['dt'])
    N_LEAPFROG = int(config['MCMC']['N_LEAPFROG'])

    datafile = config['IO']['datafile']    
    savedir  = config['IO']['savedir']    
    N_SAVE = int(config['IO']['N_SAVE'])
    N_RESTART = int(config['IO']['N_RESTART'])

    return N_GRID, L_BOX,\
            N_MCMC, dt, N_LEAPFROG,\
            datafile, savedir, N_SAVE, N_RESTART 


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
            e_r_hMpc = f['r_hMpc'][:]
            RA     = f['RA'][:]
            DEC    = f['DEC'][:]
            z_obs  = f['z_obs'][:]

    return r_hMpc, e_r_hMpc, RA, DEC, z_obs
