import numpy as np
import random as random
import networkx as nx                                            
                                                               
import pandas as pd
import torch.nn as nn 
import torch
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'runtime' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

class data():
    """Generate and save synthetic coupled Lorenz trajectories."""
    def __init__(self, args):
        self.noise_sigma = args.noise_sigma
        self.args = args
        self.N = args.N 
        self.n = args.n              
        self.T = args.T                       
        self.V = args.V           
        self.dt = args.dt
        self.ddt = args.ddt
        
        self.c = args.couple_str        
                                                             
        self.h =  (np.random.rand(self.n) - 0.5) * 0.5        
        
        self.net = self.gen_net()
    
    def gene(self):
        """Generate the synthetic data and write it to the runtime directory."""
        Xss = self.gen_data() 
        self.sav_data(Xss)
    
    def gen_net(self):
        """Build the configured coupling graph."""
        args = self.args
        if args.direc:                                                        
            net = nx.DiGraph()                                  
        else:
            net = nx.Graph() 
        net.add_nodes_from(np.arange(self.n))                             
        if args.net_nam == 'er':                       
            for u, v in nx.erdos_renyi_graph(self.n, 0.6).edges():                           
                net.add_edge(u,v,weight=random.uniform(0,1))
        elif args.net_nam == 'ba':                            
            for u, v in nx.barabasi_albert_graph(self.n, 4).edges():
                net.add_edge(u,v,weight=random.uniform(1,1)) 
        elif args.net_nam == 'rg':            
            for u, v in nx.random_regular_graph(4,self.n).edges():
                net.add_edge(u,v,weight=random.uniform(1,1))
        elif args.net_nam == 'edges':              
            edges = np.array([[0,1],[0,2],[0,3],[0,4],[1,0],[1,2],[1,3],[2,1],[2,4],[3,2]])
            weights = np.array([0.4,0.3,0.2,0.1,0.2,0.3,0.5,0.3,0.7,1.0])
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
        else:
            edges = pd.read_csv(DATA_DIR / "edges.csv").values[:,1:]
            weights = pd.read_csv(DATA_DIR / "weights.csv").values[:,1:]
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
            
        for i in range(args.n):
            swei = 0
            for j in net.neighbors(i):
                swei = swei+net.get_edge_data(i,j)['weight'] 
            for j in net.neighbors(i):
                net.edges[i,j]['weight'] = net.edges[i,j]['weight']/swei
        return(net)
    
                   
    def F(self, x, k):
        """Evaluate the intrinsic Lorenz dynamics of subsystem k."""
        f = np.zeros(x.shape)
        f[0] = -10*(x[0]-x[1])
        f[1] = 28*(1+self.h[k])*x[0] - x[1] - x[0]*x[2]
        f[2] = -8/3*x[2] + x[0]*x[1]
        return(f)
    
                       
    def G(self, x, y):
        """Evaluate the pairwise coupling dynamics."""
        g = np.zeros(x.shape)
        g[0] = -10*(y[1]-x[1])
        return(g)
    
    def gen_data(self):
        """Simulate the coupled system with Euler integration."""
        T, N, n, V, dt, ddt = self.T, self.N, self.n, self.V, self.dt, self.ddt
        net, F, G = self.net, self.F, self.G
        c = self.c
        Xs = np.zeros((T*N,n,V))
        for Ni in range(N):
                                
            x = np.zeros((T,n,V))
            x_cur = np.ones((n,V)) + 5*1e-1*np.random.rand(n,V) 
            x[0,:,:] = x_cur
            deln = int(dt/ddt)
                                
            for it in range(T*deln-1):
                for i in range(n):
                    f = F(x_cur[i,:],i)
                    g = 0
                    for j in net.neighbors(i):
                        g += G(x_cur[i,:],x_cur[j,:])*net.get_edge_data(i,j)['weight']
                    dx = (f + c*g)*ddt
                    x_cur[i,:] = x_cur[i,:] + dx
                if (it+1)%deln == 0:
                    x[int((it+1)/deln),:,:] = x_cur
            Xs[Ni*T:(Ni+1)*T,:,:] = x 
        Xss = np.zeros((T*N*V,n))
        for Vi in range(V):
            Xss[Vi*T*N:(Vi+1)*T*N,:] = Xs[:,:,Vi]
        noise = self.noise_sigma * np.random.randn(Xss.shape[0],Xss.shape[1])              
        Xss += noise
        return(Xss)
    
    def sav_data(self, Xss):
        """Persist generated trajectories and graph metadata."""
        time_point = np.arange(self.T)*self.dt
        Xss = pd.DataFrame(Xss)
        time_point = pd.DataFrame(time_point)
        Xss.to_csv(DATA_DIR / "trajectory.csv")
        Xss.to_csv(DATA_DIR / "trajectory_0.csv")           
        print("Generated CL trajectory:", DATA_DIR / "trajectory.csv")

        time_point.to_csv(DATA_DIR / "time_point.csv")
        edges = self.net.edges()
        edges = pd.DataFrame(edges())
        edges.to_csv(DATA_DIR / "edges.csv")
        
        net = self.net
        weights = []
        for u,v in net.edges():
            weights.append(net.edges[u,v]['weight'])
        weights = pd.DataFrame(weights)
        weights.to_csv(DATA_DIR / "weights.csv")
        
                                                                                 
                                                                                  
        edges_in = pd.DataFrame(np.array([[1, 1]]))
        edges_in.to_csv(DATA_DIR / "edges_in.csv")
        edges_ex = pd.DataFrame(np.array([[1],[2],[2]]))           
        edges_ex.to_csv(DATA_DIR / "edges_ex.csv")
        
        print("Data generated successfully!")
    
    











