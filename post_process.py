import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys

from fwd_PV.velocity_box import ForwardModelledVelocityBox
from fwd_PV.tools.cosmo import camb_PS
from fwd_PV.io import process_config_analysis

configfile = sys.argv[1]
N_START = int(sys.argv[2])
N_END   = int(sys.argv[3])

N_GRID, L_BOX, savedir, PROCESS_3D_V_DELTA, CALCULATE_MEAN_STD,\
            window, smoothing_scale = process_config_analysis(configfile) 

kh, pk = camb_PS()
InferenceBox = ForwardModelledVelocityBox(N_GRID, L_BOX, kh, pk, smoothing_scale, window)

if(PROCESS_3D_V_DELTA):
    for i in range(N_START, N_END):
        print(i)
        with h5.File(savedir+'/mcmc_'+str(i)+'.h5','r+') as f:
            delta_k = f['delta_k'][:]
            is_delta_3d_in_f = ('delta_3d' in f)
            is_Vr_in_f = ('Vr' in f)
            if(~is_delta_3d_in_f):
                print('Calculating delta_3d')
                delta_3d = InferenceBox.get_delta_grid(delta_k)
                f['delta_3d'] = delta_3d
            if(~is_Vr_in_f):
                print('Calculating Vr')
                Vr = InferenceBox.Vr_grid(delta_k)   
                f['Vr'] = Vr


if(CALCULATE_MEAN_STD):
    Vr_list = [] 
    for i in range(N_START, N_END):
        with h5.File(savedir+'/mcmc_'+str(i)+'.h5', 'r') as f:
            print("Reading file: mcmc_"+str(i)+".h5")
            Vr_i = f['Vr'][:]
            Vr_list.append(Vr_i)        
    Vr_list = np.array(Vr_list)
    print(Vr_list.shape)
    Vr_mean = np.mean(Vr_list, axis=0)
    Vr_std  = np.std(Vr_list, axis=0)
    with h5.File(savedir+'/mean_std.h5','w') as f:
        f['N_START'] = N_START
        f['N_END']   = N_END
        f['Vr_mean'] = Vr_mean
        f['Vr_std']  = Vr_std
