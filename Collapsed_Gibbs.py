# coding: utf-8
import numpy as np

class Collapsed_Gibbs: 
    
    #so that the counts are only computed once
    class alpha_r:
        def __init__(self, alpha):
            self.alpha = alpha
            
        def counts_vec(self,all_z):
            # c_vec_i = (αi_+􏰀\sum_{n} 1[z_mn=i]))
            c_vec = self.alpha + np.bincount(all_z,minlength =len(self.alpha))
            return c_vec
            
    
    #P(zmn|{z⇁mn},w,α,β) 
    class z_mn:
    
        def __init__(self, beta_w_mn):
            self.beta = beta_w_mn
            
        
        def sample(self, c_vec):
            #alpha_r_vec_i = (αi_+􏰀\sum_{r!=n} 1[z_mr=i]))
            alpha_r_vec = np.copy(c_vec)
            #so we can subract off the count countribution of this z_mn
            if hasattr(self,"topic"):
                last_state = self.topic
                #subtract off this document's last state
                np.put(alpha_r_vec,last_state, alpha_r_vec[last_state]-1)
            
            #compute cat probs
            probs = np.multiply(self.beta, alpha_r_vec)
            #normalize
            probs = probs/float(np.sum(probs))
            self.topic = np.nonzero(np.random.multinomial(1, probs))[0][0]
            return self.topic

    #set up gibbs sampling for a txt_ready file (given by path)
    def __init__(self, txt_ready):
        #parse the alpha vector and βwn vector 
        in_lines = open(txt_ready, 'r').readlines()
        alphas = [float(y) for y in in_lines[1].split()]
        word_probs = [[float(x) for x in in_lines[n].split()[1:]] for n in range(2,len(in_lines))]
        
        #construct the alpha_r and z_mn objects
        al = Collapsed_Gibbs.alpha_r(np.array(alphas))
        z_s =  [Collapsed_Gibbs.z_mn(np.array(wp)) for wp in word_probs]
        #save them as attributes for the sampler
        self.z = z_s
        self.c_vector = al
        
        #initialization
        self.last_z = [zmn.sample(self.c_vector.counts_vec([])) for zmn in self.z]
        
    #perform sampling
    def sample(self, n_samps=10000, burns=50):
        samples = []
        for samp in range(n_samps):        
            #compute the counts + alphas
            counts_t = self.c_vector.counts_vec(self.last_z)
            #sample all the z_mn
            z_t = [zmn.sample(counts_t) for zmn in self.z]
            self.last_z = np.copy(z_t)
            #add samples
            samples.append(np.copy(z_t))    
        return samples[burns:]  