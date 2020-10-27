import pandas as pd
import numpy as np
import h5py as h5

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
