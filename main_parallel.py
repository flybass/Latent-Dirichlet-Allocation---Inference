from mean_Field import *
from Collapsed_Gibbs import *
from Gibbs import *
import matplotlib.pyplot as plt

import glob
import os
import re

folder_path  = os.path.join('..','ps4_data')
files =  glob.glob( folder_path + "/*ready")


def E_collapsed(coll_path, alphas):
	counts = [np.bincount(c_p, minlength =len(alphas)) for c_p in coll_path]
	#compute sums at each point in time 
	counts_t = []
	for t in range(len(coll_path)):
		if t>0:
			counts_t.append(counts_t[t-1] + counts[t])
		else:
			counts_t.append(counts[t])

	#compute expectations
	expectations = []
	for t in range(len(coll_path)):
		T = t+1
		counts_sum = counts_t[t]
		numerator = T*np.array(alphas) + counts_sum
		denominator = T*(np.sum(alphas) + len(coll_path[0]))
		expectations.append(numerator* 1./denominator)
	return expectations

def E_others(theta_path):
	#make means iteratively
	means_array = []
	for t in range(len(theta_path)):
		if t>0:
			new_arr = (means_array[t-1]*(t) + theta_path[t])*1./(t+1)
		else:
			new_arr = theta_path[t]
		means_array.append(new_arr)
	return means_array

def l2_error(estimate,g_t):
	error = np.linalg.norm(g_t - estimate,2)**2
	return error


def run_and_save(f):
	#perform sampling
	g_path=Gibbs(f).sample(10000)
	collapsed_path = Collapsed_Gibbs(f).sample(10000)
	#a little different for this case since we only run
	#till convergence
	mf =mean_Field(f)
	mf.va_inf()

	#Compute E[theta]
	#gibbs theta
	g_thetas = [g_p[0] for g_p in g_path]
	#Compute expectations at each point in time for gibbs
	exp_g = E_others(g_thetas)
	#expectations for collapsed
	exp_collapsed = E_collapsed(collapsed_path, mf.alpha)

	#expectations for mean_field
	mf_thetas = [np.array(y) for y in mf.thetas()]
	exp_mf = E_others(mf_thetas)

	#GROUND TRUTH
	g_t = exp_collapsed[-1]
	#l2_errors
	error_g = [l2_error(g, g_t) for g in exp_g]
	error_coll = [l2_error(h, g_t) for h in exp_collapsed]
	error_mf = [l2_error(i, g_t) for i in exp_mf]
	last_v = error_mf[-1]
	error_mf = error_mf + [last_v for i in range(9950-len(error_mf))]

	plt.title(re.search("(nips.*)\.txt", f).group(1))
	p1,= plt.plot(error_g, label="Gibbs LDA")
	p2, = plt.plot(error_coll, label="Collapsed Gibbs LDA")
	p3, = plt.plot(error_mf, label="Mean-Field")
	plt.xlabel("Iterations")
	plt.ylabel("L2 Error")
	plt.figlegend([p1,p2,p3],["Gibbs LDA", "Collapsed Gibbs LDA","Mean-Field"], 'right')
	plt.ylim([0,.005])
	plt.savefig(re.search("(nips.*)\.txt", f).group(1)+".png", format="png")
	plt.clf()


import multiprocessing
p = multiprocessing.Pool(multiprocessing.cpu_count())	
p.map(run_and_save,files)



