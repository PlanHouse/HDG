                                                 
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
import pandas as pd
import joblib 

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path

from models.reservoir_model_HoGRC import Reservoir
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use('Agg')                           

RUNTIME_DIR = Path(__file__).resolve().parent.parent / 'runtime'
DATA_DIR = RUNTIME_DIR / 'data'
READOUT_DIR = RUNTIME_DIR / 'readouts'
READOUT_DIR.mkdir(parents=True, exist_ok=True)



class model():
    """Reservoir-computing evaluator used by the screening workflow."""
    def __init__(self, args, lag_idx=None):
        self.args = args 
        self.Xsn0, self.Xsn, self.time_point, self.edges_in, self.edges_ex, self.edges_out = self.read_data()
        
                                                           
        self.ntr = int(600)
                             
        self.loss_f = torch.nn.MSELoss()
        self.epochs = args.epochs
        self.batchs = args.batchs
        self.device = args.device
        self.base_seed = args.base_seed
        self.reservoir = self.RC_param(lag_idx)

    def _plot_results(self, test_y, pred_y, path, lag_i, error):
        """Save a diagnostic prediction plot for one lag."""
        
        plt.figure(figsize=(15, 5))

              
        plt.plot(test_y[-1000:], label="True Values")
        plt.plot(pred_y[-1000:], '--', label="Predictions")
        plt.xlabel("DMU_origin04 Time Steps")
        plt.ylabel("Values")
        plt.title(f"Prediction Results_lag{lag_i+1}   error:{error}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

                           
        save_path = f"{path}/prediction_results_lag{lag_i+1}.png"                     
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()               
    
    def read_data(self):
        """Load runtime trajectories and reshape them for reservoir evaluation."""
        args = self.args
        Xs = pd.read_csv(DATA_DIR / "trajectory.csv").values[:,1:].transpose()             
        Xss = np.zeros((args.n, args.N*args.T, args.V))                                               
        Xsn = np.zeros((args.N, args.n, args.T, args.V))                                                
        for i in range(args.V):
            Xss[:,:,i] = Xs[:,i*args.N*args.T:(i+1)*args.N*args.T]
        for i in range(args.N):
            Xsn[i,:,:,:] = Xss[:,i*args.T:(i+1)*args.T,:]
        time_point = pd.read_csv(DATA_DIR / "time_point.csv").values[:, 1:]         
        edges_out = pd.read_csv(DATA_DIR / "edges.csv").values[:, 1:]
        edges_in = pd.read_csv(DATA_DIR / "edges_in.csv").values[:, 1:]
        edges_ex = pd.read_csv(DATA_DIR / "edges_ex.csv").values[:, 1:]
        Xsn = torch.tensor(Xsn).float()
        Xsn = Xsn + args.ob_noise * np.random.randn(Xsn.shape[0],Xsn.shape[1],Xsn.shape[2],Xsn.shape[3])          

        Xs0 = pd.read_csv(DATA_DIR / "trajectory_0.csv").values[:,1:].transpose()             
        Xss0 = np.zeros((args.n, args.N*args.T, args.V))                                               
        Xsn0 = np.zeros((args.N, args.n, args.T, args.V))                                                
        for i in range(args.V):
            Xss0[:,:,i] = Xs0[:,i*args.N*args.T:(i+1)*args.N*args.T]
        for i in range(args.N):
            Xsn0[i,:,:,:] = Xss0[:,i*args.T:(i+1)*args.T,:]
        Xsn0 = torch.tensor(Xsn0).float()
        Xsn0 = Xsn0 + args.ob_noise * np.random.randn(Xsn0.shape[0],Xsn0.shape[1],Xsn0.shape[2],Xsn0.shape[3])          

        Xsn0 = Xsn0[:, :, 2000:2900, :]
        Xsn = Xsn[:, :, 2000:2900, :]
        time_point = time_point[2000:2900, :]
                                    
                                  
                                          

        return(Xsn0, Xsn, time_point, edges_in, edges_ex, edges_out)
    def RC_param(self, lag_idx=None):
        """Construct the reservoir with the current lag-specific encoding."""
        args = self.args
        edges_in = self.edges_in 
        edges_ex = self.edges_ex 
        edges_out = self.edges_out
        reservoir = Reservoir(n_internal_units = args.n_internal_units,
                              spectral_radius = args.spectral_radius,
                              leak = args.leak,
                              connectivity = args.connectivity,
                              input_scaling = args.input_scaling,
                              noise_level = args.noise_level,
                              base_seed = self.base_seed,
                              lag_idx = lag_idx,
                              edges_in = edges_in,
                              edges_ex = edges_ex,
                              edges_out = edges_out,
                              args = args)
        return(reservoir)
    
    def train(self, lag_pre):
        """Fit ridge-regression readouts from reservoir states."""                    
                           
        Xsn0, Xsn, args, ntr =self.Xsn0, self.Xsn, self.args, self.ntr             
        reservoir = self.reservoir
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
                                                                      
        Ni = 0
        warm_up = args.warm_up 
        X_train = res_states[Ni,warm_up:ntr,:]
        Y_train = (Xsn0[Ni, :, warm_up + lag_pre: ntr + lag_pre, :].numpy()
                   - Xsn0[Ni, :, warm_up: ntr, :].numpy()) / args.dt                                                   

        for ni in range(args.n):
            out_weigth=np.zeros((int(res_states.shape[-1]/ args.V),args.V))
            for Vi in range(args.V):
                X = X_train[:,(ni*args.V+Vi)*args.n_internal_units:(ni*args.V+Vi+1)*args.n_internal_units]
                Y = Y_train[ni,:,Vi]
                readout = Ridge(alpha=args.alpha)
                readout.fit(X,Y)
                joblib.dump(readout, READOUT_DIR / ('readout' + str(ni) + '_' + str(Vi) + '.pkl'))                                    
                out_weigth[:,Vi]=readout.coef_
                                                   
        return (out_weigth)
                                    

    def evalue1(self,lag_i, lag_pre):
        """Generate one-step predictions and return the absolute error."""                                   
                              
        reservoir = self.reservoir
        args, Xsn, Xsn0 = self.args, self.Xsn, self.Xsn0
                                                     
        N, n, T, V = args.N, args.n, self.Xsn.shape[2], args.V
        warm_up = args.warm_up
        Ni = 0
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)                 

        readout_dict = {}
        for ni in range(n):
            for Vi in range(V):
                readout_dict[(ni, Vi)] = joblib.load(READOUT_DIR / ('readout' + str(ni) + '_' + str(Vi) + '.pkl'))
        
        preds = np.zeros((N, n, T-warm_up - lag_pre + 1, V))
                                                 
        for ni in range(n):
            for Vi in range(V):
                readout = readout_dict[(ni, Vi)]
                                                                                                    
                for i in range(T - warm_up - lag_pre + 1):
                                              
                    X = res_states[:, i+ warm_up-1, (ni*V+Vi)*args.n_internal_units:(ni*V+Vi+1)*args.n_internal_units]                 

                    preds[Ni, ni, i, Vi] = readout.predict(X) * args.dt + Xsn0[:, ni, i + warm_up - 1, Vi].numpy()                 
        error = (Xsn0[:, :, warm_up + lag_pre - 1: T, :] - preds).numpy()
        test_y = Xsn0[:, :, warm_up + lag_pre - 1: T, :].numpy().reshape(-1, args.V)
        pred_y = preds.reshape(-1, args.V)
                                           
                                                                                 
                                        
        return (preds, error)
