# coding: utf-8
import numpy as np
import math as m
import scipy.special as sps
import copy as cp

class mean_Field:
    
    def __init__(self, txt_ready):
        #parse the alpha vector and βwn vector 
        in_lines = open(txt_ready, 'r').readlines()
        alphas = [float(y) for y in in_lines[1].split()]
        word_probs = [[float(x) for x in in_lines[n].split()[1:]] for n in range(2,len(in_lines))]
        
        #save as attributes for va_inf
        self.alpha = alphas
        self.beta = word_probs
    
    #set the gammas and phis
    def va_inf(self):
        #initialization
        #initialize φ_ni := 1/k for all i and n
        #initialize γ_i :=α_i+N/k for all i
        K = len(self.alpha)
        N = len(self.beta)
        phi = np.array([1./K]*K)
        phis = [phi for i in range(N)]
        gamma = np.array([al + N/float(K) for al in self.alpha])
        
        #last values
        last_phi = cp.copy(phis)
        last_gamm = cp.copy(gamma)
        
        iters = []
        converged = False
        while not(converged):
            for n in range(N):
                for k in range(K):
                    phis[n][k] = self.beta[n][k] * m.exp(sps.psi(gamma[k]))
                norm_term = sum(phis[n])
                phis[n] = [x/float(norm_term) for x in phis[n]]
            gamma = np.array(self.alpha) + np.sum(phis, axis=0)
            converged = self.converged_check(phis, last_phi,gamma, last_gamm )
            last_phi = phis
            last_gamm = gamma
            iters.append([np.copy(phis), np.copy(gamma)])
        self.va = iters
        return iters
        
    def thetas(self):
        thetas = []
        for it in range(len(self.va)):
            gams = self.va[it][1]
            E_theta = [g/float(np.sum(gams)) for g in gams]
            thetas.append(E_theta)
        return thetas
              
        
    def converged_check(self, phis, last_phi,gamma, last_gamm ):
        phi_0= np.array(phis)
        phi_1 = np.array(last_phi)
        phi_dist = np.linalg.norm(phi_0 - phi_1,2)
        gam_dist = np.linalg.norm(gamma - last_gamm,2)
        return (abs(phi_dist) + abs(gam_dist)==0)