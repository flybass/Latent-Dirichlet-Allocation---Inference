# coding: utf-8
import numpy as np

class Gibbs: 
    
    #p(θ_i|θ_⇁i, z, w, α, β)
    class theta_i:
        #the alphas are listed in order in order 1...k
        def __init__(self, alphas):
            self.alphas = np.copy(alphas)
    
        #z_i is the set of z_i
        def sample(self, z_i):
            new_alphas = self.alphas + np.bincount(z_i,minlength =len(self.alphas))
            return np.random.dirichlet(new_alphas)

    #p(zmn|{θ},{z⇁mn},w,α,β)   
    class z_mn:
    
        def __init__(self, beta_w_mn):
            self.beta = beta_w_mn
        
        def sample(self, theta):
            probs = np.multiply(theta, self.beta)
            probs = probs/float(np.sum(probs))
            return np.nonzero(np.random.multinomial(1, probs))[0][0]

    #set up gibbs sampling for a txt_ready file (given by path)
    def __init__(self, txt_ready):
        #parse the alpha vector and βwn vector 
        in_lines = open(txt_ready, 'r').readlines()
        alphas = [float(y) for y in in_lines[1].split()]
        self.alphas = alphas
        word_probs = [[float(x) for x in in_lines[n].split()[1:]] for n in range(2,len(in_lines))]
        
        #construct the theta and z_mn objects
        theta_m = Gibbs.theta_i(np.array(alphas))
        z_s =  [Gibbs.z_mn(np.array(wp)) for wp in word_probs]
        #save them as attributes for the sampler
        self.theta = theta_m
        self.z = z_s
        
        #initialization
        self.last_theta = np.random.dirichlet(np.array(alphas))
        self.last_z = [zmn.sample(self.last_theta) for zmn in self.z]
        
    #perform sampling
    def sample(self, n_samps=10000, burns=50):
        samples = []
        for samp in range(n_samps):        
            #sample theta
            theta_t = self.theta.sample(self.last_z)
            self.last_theta = np.copy(theta_t)
            #sample all the z_mn
            z_t = [zmn.sample(theta_t) for zmn in self.z]
            self.last_z = np.copy(z_t)
            #add samples
            samples.append([np.copy(theta_t), np.copy(z_t)])    
        return samples[burns:]  