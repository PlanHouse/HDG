# Algorithm 1 workflow: lag-wise RC screening and weight generation.
                                                                                                              
                                                                                                  
                                                                              
                                                                                                              
import argparse
import numpy as np
import importlib
import pandas as pd
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PACKAGE_ROOT / 'runtime'
DATA_DIR = RUNTIME_DIR / 'data'
WEIGHTS_DIR = RUNTIME_DIR / 'weights'

        

                                                                
                                                                
                                                                
def args():
    """Create the fixed configuration used by the screening experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation (e.g., "cpu", "cuda").')
    parser.add_argument('--model_ind', type=str, default='HoGRC', help='Method identifier.')
    parser.add_argument('--data_ind', type=str, default='CL', help='Data identifier.')            
    parser.add_argument('--net_nam', type=str, default='er', help='Network name.')
    parser.add_argument('--direc', type=bool, default=True, help='Direction of network (True for directed).')
    parser.add_argument('--nj', type=int, default=0, help='the nj-th variable.')

                                     
    parser.add_argument('--N', type=int, default=1, help='Number of samples.')
    parser.add_argument('--n', type=int, default=1, help='Number of subsystem')
    parser.add_argument('--T', type=int, default=5000, help='Number of data points.')
    parser.add_argument('--V', type=int, default=3, help='Number of dimension in a subsystem.')
    parser.add_argument('--dt', type=float, default=0.02, help='Sampling time step size.')
    parser.add_argument('--ddt', type=float, default=0.001, help='Simulating time step size.')
    parser.add_argument('--couple_str', type=float, default=1, help='Coupling strength.')
    parser.add_argument('--noise_sigma', type=float, default=0.0, help='Noise strength.')
    parser.add_argument('--ob_noise', type=float, default=0.0, help='Observational noise strength.')
    parser.add_argument('--qtr', type=float, default=0.6, help='Training ratio.')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold value for VPS index.')

                      
    parser.add_argument('--pre_step', type=int, default=1, help='the length of predict step.')
    parser.add_argument('--warm_up', type=int, default=50, help='Warm-up period for reservoir.')
    parser.add_argument('--n_internal_units', type=int, default=50, help='Number of internal units in reservoir.')
    parser.add_argument('--spectral_radius', type=float, default=0.85,
                        help='Spectral radius of the reservoir adjacency matrix.')
    parser.add_argument('--leak', type=float, default=0.05, help='Leak rate of the reservoir.')
    parser.add_argument('--sigma', type=float, default=1, help='Dynamical bias for reservoir.')
    parser.add_argument('--connectivity', type=float, default=0.1, help='Connectivity of the reservoir.')
    parser.add_argument('--input_scaling', type=float, default=0.1, help='Input scaling factor for reservoir.')
    parser.add_argument('--noise_level', type=float, default=0, help='Noise level for reservoir.')
    parser.add_argument('--alpha', type=float, default=10 ** (-8),
                        help='Regularization coefficient (ridge regression).')

                                          
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batchs', type=int, default=300, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 penalty).')
    parser.add_argument('--base_seed', type=int, default=2025,
                        help='Base random seed for reservoir initialization.')

                 
    parser.add_argument('--weight_position', type=str, default=str(WEIGHTS_DIR),
                        help='Directory for generated reservoir weights.')


    args = parser.parse_args(args=[])
    return (args)


args = args()
for output_dir in (DATA_DIR, WEIGHTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
model_ind = 'HoGRC'

                                                                
                                                                
                                                                

seed = 2
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


dataset = importlib.import_module('dataset.Data_' + args.data_ind)
data = dataset.data

def gen_data(data):
    """Generate the synthetic trajectory used by the RC evaluator."""
    dat = data(args)
    dat.gene()

gen_data(data)


                                                           
                                                                                                                                                  

                                                                
                                                                          
                                                                
def gen_ein(candi):
    """Encode a candidate family as the RC input-structure mask."""            
    edges_in = []
    for node, neigh in zip(candi.keys(), candi.values()):                            
        j = va[node]
        for hi in range(len(neigh)):
            comp = neigh[hi]
            decimal = 0
            for ci in comp:
                decimal = decimal + va[ci]
            edges_in.append([j, decimal])
    edges_in = pd.DataFrame(np.array(edges_in))
    edges_in.to_csv(DATA_DIR / "edges_in.csv")                   
    return (edges_in)


def HORC(candi, lag_data=None, lag_idx=None):
    """Train and evaluate the RC model for one candidate structure."""                                           
                                                                          

    seed = 111         
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)                 

    print(np.random.random(2))

    mod = importlib.import_module('models.Model_' + model_ind)
    model = mod.model                     
    lag_pre = args.pre_step                         
    ntr = int(args.T * args.qtr)
    experiment = model(args, lag_idx)          
    experiment.train(lag_pre)        
    preds, error = experiment.evalue1(lag_data, lag_pre)                 
    print("  --Total error:", np.mean(np.abs(error)))
    er = np.mean(np.abs(error[:, 0, :ntr - args.warm_up, Vu]))
                                 
                                            
                         
    print(f"  --Candidate neighbors: {candi[u]}, Error_{u}: {er}  \n")
    return (er)


def RD(candi_comp, ci):
    """Return one-variable reductions of the specified candidate member."""      
    rd_ci = []
    for i in range(len(ci)):
        mci = ci.copy()
        mci.remove(ci[i])
        ind = 1
        for ccpi in candi_comp:
            if mci[0] in ccpi:
                ind = 0
        if ind:
            rd_ci.append(mci)
    return (rd_ci)

def evaluate_and_update(candi, e1, epsilon, ho):
    """Accept the candidate when its error remains within the tolerance."""
    
    opt = False
    hoo = ho.copy()
    edges_in = gen_ein(candi)
    e2 = HORC(candi, lag_data=0, lag_idx=None)
    hous.append(candi[u].copy())
    errors.append(e2)
    if e2 - e1 < epsilon:
        opt = True
        hoo = candi.copy()
        e1 = e2
    return opt, e1, hoo, errors

def try_removal_from_complexes(complexes, e1, epsilon, ho):
    """Evaluate all candidate families created by removing one member."""
    
    opt = False
    hoo = ho.copy()
    for i in range(len(complexes)):
        temp = complexes[:i] + complexes[i + 1:]
        candi[u] = temp
        improved, new_e1, updated_ho, errors = evaluate_and_update(candi, e1, epsilon, ho)
        if improved:
            opt = True
            hoo = updated_ho.copy()
            e1 = new_e1
    return opt, e1, hoo, errors

def try_split_subsets(complexes, e1, epsilon, ho):
    """Evaluate families obtained by reducing one variable from a member."""
    
    opt = False
    hoo = ho.copy()
    for i, group in enumerate(complexes):
        for j in range(len(group)):
            reduced = list(set(group) - {group[j]})
            if reduced == []:
                temp_complexes = complexes[:i] + complexes[i + 1:]
            else:
                temp_complexes = complexes[:i] + [reduced] + complexes[i + 1:]
                  
            temp_complexes = [sublist for index, sublist in enumerate(temp_complexes) if
                    tuple(sublist) not in set(tuple(temp_complexes[t]) for t in range(index))]           
                                                                                            
            candi[u] = temp_complexes
            improved, new_e1, updated_ho, errors = evaluate_and_update(candi, e1, epsilon, ho)
            if improved:
                opt = True
                hoo = updated_ho.copy()
                e1 = new_e1
    return opt, e1, hoo, errors

start_time = time.time()

print("Starting Algorithm 1 screening for target x.")

ho = {'x': [['x', 'y', 'z']], 'y': [['x', 'y', 'z']], 'z': [['x', 'y', 'z']]}                     
va = {'x': 1, 'y': 2, 'z': 4}

u = 'x'                         
Vu = int(np.log2(va[u]))
epsilon = 5 * 1e-7           
not_delete = []
not_reduce = []
hous = []          
errors = []                

candi = ho.copy()            
edges_in = gen_ein(candi)
e1 = HORC(candi, lag_data=0, lag_idx=None)

hous.append(ho[u].copy())
errors.append(e1)

circle = 1
Houxuan = ho.copy()
Houxuan[u] = []

while circle:
    comp = ho[u]
    candi_comp = comp.copy()
    ci = comp[0]

    if len(candi_comp) == 1 and len(ci) == 1:
        Houxuan[u] = ho[u].copy()
        break

    elif len(candi_comp) == 1 and len(ci) > 1:
                     
        candi_comp.remove(ci)
        candi_comp.extend(RD(candi_comp, ci))
        candi[u] = candi_comp
        improved, e1, new_ho, errors = evaluate_and_update(candi, e1, epsilon, ho)
        if improved:
            ho = new_ho
            continue
                            
        improved, e1, new_ho, errors = try_removal_from_complexes(candi_comp, e1, epsilon, ho)
        if improved:
            ho = new_ho
            continue
        break       

    elif len(candi_comp) > 1:
                        
        improved, e1, new_ho, errors = try_removal_from_complexes(candi_comp, e1, epsilon, ho)
        if improved:
            ho = new_ho
            continue

                         
        improved, e1, new_ho, errors = try_split_subsets(candi_comp, e1, epsilon ,ho)
        if improved:
            ho = new_ho
            continue
        break       

print("###################################################")
print("One-step Prediction Error with different candidate neighbors:")
print(errors)
print("The optimal higher-order neighbors of z are", e1, ho[u])
H0_neighbors=[]
H0_neighbors.append(ho[u].copy())         
print("###########################################################################################\n")

       
for lag_i in range(5):                               
    ho = {'x': [['x', 'y', 'z']], 'y': [['x', 'y', 'z']], 'z': [['x', 'y', 'z']]}                     
    va = {'x': 1, 'y': 2, 'z': 4}

    Vu = int(np.log2(va[u]))
    print(f"\nStarting lag {lag_i + 1} for target {u}.")
                    
    Xs = pd.read_csv(DATA_DIR / "trajectory_0.csv").values[:, 1:]      
    data_Xs = Xs          
    Len_Xs = int(Xs.shape[0] / args.V)        
    Xs_NEW = np.zeros((Len_Xs, args.V))          

                                                                                                                                                       
    Xs_NEW[lag_i+1:, :] =np.concatenate(
        (Xs[:Len_Xs - (lag_i + 1)], Xs[Len_Xs:Len_Xs * 2 - (lag_i + 1)], Xs[Len_Xs * 2:3 * Len_Xs - (lag_i + 1)]),
        axis=1)          
    Xs_NEW = Xs_NEW.T.reshape(-1, 1)
    Xs_NEW = pd.DataFrame(Xs_NEW)
    temp_name = lag_i+1
                                                                   
    Xs_NEW.to_csv(DATA_DIR / "trajectory.csv")      
                    

    epsilon = 5 * 1e-7           
    not_delete = []
    not_reduce = []
    hous = []          
    errors = []                

    candi = ho.copy()            
    edges_in = gen_ein(candi)              
    e1 = HORC(candi, lag_data=lag_i+1, lag_idx=None)                 

    hous.append(ho[u].copy())                 
    errors.append(e1)                    

    circle = 1
    Houxuan = ho.copy()
    Houxuan[u] = []

    while circle:
        comp = ho[u]
        candi_comp = comp.copy()
        ci = comp[0]

        if len(candi_comp) == 1 and len(ci) == 1:
            Houxuan[u] = ho[u].copy()
            break

        elif len(candi_comp) == 1 and len(ci) > 1:
                     
            candi_comp.remove(ci)
            candi_comp.extend(RD(candi_comp, ci))
            candi[u] = candi_comp
            improved, e1, new_ho, errors = evaluate_and_update(candi, e1, epsilon, ho)
            if improved:
                ho = new_ho
                continue
                            
            improved, e1, new_ho, errors = try_removal_from_complexes(candi_comp, e1, epsilon, ho)
            if improved:
                ho = new_ho
                continue
            break       

        elif len(candi_comp) > 1:
                        
            improved, e1, new_ho, errors = try_removal_from_complexes(candi_comp, e1, epsilon, ho)
            if improved:
                ho = new_ho
                continue

                         
            improved, e1, new_ho, errors = try_split_subsets(candi_comp, e1, epsilon, ho)
            if improved:
                ho = new_ho
                continue
            break       

    print("One-step Prediction Error with different candidate neighbors:")
    print(errors)
    print(f"The optimal higher-order neighbors of are {ho[u]}, z_error:{e1}")
    print(f"Finished lag {lag_i + 1} for target {u}.")
    H0_neighbors.append(ho[u].copy())

H0_neighbors_x=H0_neighbors



        
end_time = time.time()

                                                                                            
print("\nGenerating lag-specific input and recurrent weights.")

                                             
HN = list(range(len(H0_neighbors)))
args.n_internal_units = 50           
for lag_i in HN:
                    
    Xs = pd.read_csv(DATA_DIR / "trajectory_0.csv").values[:, 1:]      
    data_Xs = Xs          
    Len_Xs = int(Xs.shape[0] / args.V)        
    Xs_NEW = np.zeros((Len_Xs, args.V))          

                                                                                                                                                       
    Xs_NEW[lag_i + 1:, :] = np.concatenate(
        (Xs[:Len_Xs - (lag_i + 1)], Xs[Len_Xs:Len_Xs * 2 - (lag_i + 1)], Xs[Len_Xs * 2:3 * Len_Xs - (lag_i + 1)]),
        axis=1)          
    Xs_NEW = Xs_NEW.T.reshape(-1, 1)
    Xs_NEW = pd.DataFrame(Xs_NEW)
    temp_name = lag_i
                                                                   
    Xs_NEW.to_csv(DATA_DIR / "trajectory.csv")      
                    

    for vii in ho.keys():
        candi[vii] = H0_neighbors_x[lag_i]
                                                             
                                                             
                                                             
    edges_in = gen_ein(candi)              
    e1 = HORC(candi, lag_data=lag_i, lag_idx=lag_i)                                                                        
    print(f"  --total Candidate neighbors: {candi} \n")
print(f"Execution time: {end_time - start_time:.6f} seconds")


