import pandas as pd
import numpy as np
import jax.numpy as jnp
import h5py as h5
import configparser

def config_Pk(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    savedir  = config['IO']['savedir']

    Om_min = float(config['Pk']['Om_min'])
    Om_max = float(config['Pk']['Om_max'])
    
    Om_size = int(config['Pk']['Om_size'])

    return savedir, Om_min, Om_max, Om_size

 
def process_config_analysis(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    N_GRID = int(config['BOX']['N_GRID'])
    L_BOX  = float(config['BOX']['L_BOX'])
    Pk_type = config['BOX']['Pk_type']

    savedir  = config['IO']['savedir']

    PROCESS_3D_V_DELTA = bool(config['ANALYSIS']['process_3d_v_delta'].lower() == "true")
    CALCULATE_MEAN_STD = bool(config['ANALYSIS']['calculate_mean_std'].lower() == "true")
   
    window = config['BOX']['window']
    smoothing_scale = float(config['BOX']['smoothing_scale']) 
    
    return N_GRID, L_BOX, Pk_type, savedir, PROCESS_3D_V_DELTA, CALCULATE_MEAN_STD, window, smoothing_scale

def process_config(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    N_GRID = int(config['BOX']['N_GRID'])
    L_BOX  = float(config['BOX']['L_BOX'])
    likelihood = config['BOX']['likelihood']
    sample_cosmology = bool(config['COSMOLOGY_SAMPLING']['sample_cosmology'].lower()=="true")
    sample_sigv = bool(config['BOX']['sample_sigv'].lower()=="true")
    window = config['BOX']['window']
    Pk_type = config['BOX']['Pk_type']
    try:
        fix_density = bool(config['COSMOLOGY_SAMPLING']['fix_density'].lower()=="true")
    except:
        fix_density = False
    try:
        smoothing_scale = float(config['BOX']['smoothing_scale']) 
    except:
        smoothing_scale = 0. 

    N_MCMC = int(config['MCMC']['N_MCMC'])
    dt     = float(config['MCMC']['dt'])
    N_LEAPFROG = int(config['MCMC']['N_LEAPFROG'])

    datafile = config['IO']['datafile']    
    savedir  = config['IO']['savedir']    
    N_SAVE = int(config['IO']['N_SAVE'])
    N_RESTART = int(config['IO']['N_RESTART'])

    if(fix_density):
        true_density_path = config['COSMOLOGY_SAMPLING']['true_density_path']
    else:
        None
    return N_GRID, L_BOX, likelihood, sample_cosmology, sample_sigv, window, smoothing_scale, Pk_type,\
            N_MCMC, dt, N_LEAPFROG,\
            datafile, savedir, N_SAVE, N_RESTART,\
            fix_density, true_density_path 


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
     
def get_true_density(true_density_path, L, N):
    l = L / N
    dV = l**3
    V = L**3 
    dens = np.load(true_density_path)
    print("dens mean-std: %2.4f, %2.4f"%(np.mean(dens), np.std(dens)))
    delta_k_complex = dV / V * np.fft.rfftn(dens)
    delta_k = np.array([delta_k_complex.real, delta_k_complex.imag])
    delta_k[:,0,0,0] = 0.
    return jnp.array(delta_k)
