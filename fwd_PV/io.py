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
    try:
        coord_system_box = config['BOX']['coord']
    except:
        print('No coord system found...Using galactic coordinate as default...')
        coord_system_box = 'galactic'

    return N_GRID, L_BOX, likelihood, coord_system_box

def config_mcmc(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
   
    N_MCMC = int(config['MCMC']['N_MCMC'])
    dt     = float(config['MCMC']['dt'])
    N_LEAPFROG = int(config['MCMC']['N_LEAPFROG'])
    try:
        sample_scale = bool(config['MCMC']['sample_scale'].lower()=="true")
    except:
        sample_scale = False
    
    return N_MCMC, dt, N_LEAPFROG, sample_scale

def config_io(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    savedir  = config['IO']['savedir']    
    N_SAVE = int(config['IO']['N_SAVE'])
    N_RESTART = int(config['IO']['N_RESTART'])

    return savedir, N_SAVE, N_RESTART 

def config_data(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    N_CAT  = int(config['DATA']['N_CAT'])    
    data_arr = None
    data_len = []
    for i in range(N_CAT):
        datafile = config['DATA']['datafile%d'%(i)]
        PV_data = process_datafile(datafile, 'h5')
        if data_arr is None:
            data_arr = PV_data
        else: 
            data_arr = np.hstack([data_arr, PV_data])
        data_len.append(PV_data.shape[1])
    print("data_arr.shape: "+str(data_arr.shape))
    print("data_len: "+str(data_len))
    return N_CAT, data_arr, data_len

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
    PV_data = np.array([r_hMpc, e_r_hMpc, RA, DEC, z_obs])
    return PV_data

def config_fwd_lkl(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    N_GRID = int(config['FWD_LKL']['N_BOX_sim'])
    L_BOX  = float(config['FWD_LKL']['L_BOX_sim'])
    
    try:
        coord_system  = config['FWD_LKL']['coord_system']
    except:
        coord_system = "equatorial"
    R_lim  = float(config['FWD_LKL']['R_lim'])
    density_data = config['FWD_LKL']['density_file']
    print("Loading density data from "+density_data)
    delta_grid = np.load(density_data)
    
    return delta_grid, L_BOX, N_GRID, coord_system, R_lim
          
def write_save_file(i, N_SAVE, savedir, delta_k, ln_prob, scale):
    print('=============')
    print('Saving file...')
    print('=============')
    j = i//N_SAVE
    with h5.File(savedir + '/mcmc_'+str(j)+'.h5', 'w') as f:
        f['delta_k'] = delta_k
        f['ln_prob'] = ln_prob
        f['scale']   = scale
        
def write_restart_file(savedir, delta_k, i, scale):
    print("Saving restart file...")
    with h5.File(savedir+'/restart.h5', 'w') as f:
        f['delta_k'] = delta_k
        f['N_STEP']  = i
        f['scale']   = scale
