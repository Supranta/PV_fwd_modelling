import pandas as pd
import numpy as np

def process_datafile(datafile):
    df = pd.read_csv(datafile)

    r_hMpc = np.array(df['r_hMpc'])
    RA = np.array(df['RA'])
    DEC = np.array(df['DEC'])
    z_obs = np.array(df['z_obs'])

    return r_hMpc, RA, DEC, z_obs
