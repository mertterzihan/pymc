'''
Implementation of HD-LDA model that has been described in Distributed Algorithms 
for Topic Models by D. Newman, et al. (2009).
'''
from random import random
total_topics = 10
# Seeds for global parameters, phi and beta, to enable synchronization between the jobs
phi_seeds = [int(random()*1000000) for k in xrange(total_topics)]
beta_seeds = [int(random()*1000000) for k in xrange(total_topics)]

def model_function(data, global_param):
	from pymc import Dirichlet, Categorical, Gamma, Lambda, Container, Model
	import numpy as np

	total_partitions = 4
	total_words = 7491
	total_vocab = 78

	a = float((total_partitions-1)*total_words)/(total_partitions*total_topics)
	b = 1.0
	c = 2.0
	d = 0.1
	gamma_prior = 2.0/total_topics

	gamma_vector = [gamma_prior for v in xrange(total_vocab)]

	# Topic dependent strength parameter
	beta = Container([Gamma(name='beta_%i' % k, 
							alpha=a, 
							beta=1/b,
							value=a*b) for k in xrange(total_topics)])

	# Global word topic-distribution
	global_phi = Container([Dirichlet('global_phi_%i' % k, 
									 theta=gamma_vector,
									 value=(gamma_vector/np.sum(gamma_vector))[:-1]) for k in xrange(total_topics)])

	phi_param = [Lambda('phi_param_%i' % k, 
						lambda global_phi=global_phi, beta=beta: np.multiply(np.append(global_phi[k], 1.0-np.sum(global_phi[k])), beta[k]), 
						trace=False) for k in xrange(total_topics)]

	# Local word-topic distribution
	local_phi = Container([Dirichlet('local_phi_%i' % k, 
									 theta=phi_param[k], 
									 value=[1.0/total_vocab]*(total_vocab-1)) for k in xrange(total_topics)])

	# Local prior parameter for document-topic distribution
	alpha = Gamma(name='alpha', alpha=c, beta=1/d, value=c*d)

	alpha_vector = Lambda('alpha_vector', 
						  lambda alpha=alpha: [alpha for k in xrange(total_topics)],
						  trace=False)

	local_docs = list()
	# Given the data as a list of strings (lines), structure it in such a way that it can be used by the below model
	for doc_number,line in enumerate(data):
		document_data = line.split(',')
		words = document_data[1].split(' ')
		words = map(int, words)
		local_docs.append((int(document_data[0]), words))

	theta_value = [1.0/total_topics for i in xrange(total_topics)]

	# Document-topic distribution
	theta = Container([Dirichlet('theta_%i' % local_docs[i][0], 
								 theta=alpha_vector,
								 value=theta_value[:-1]) for i in xrange(len(local_docs))])

	# Topic assignments for each word
	z = Container([Categorical('z_' + str(doc[0]), 
							   p=theta[n], 
							   size=len(doc[1]), 
							   verbose=0) for n, doc in enumerate(local_docs)])

	local_phi_lambda = [Lambda('local_phi_lambda_%i' % k,
							   lambda local_phi=local_phi: np.append(local_phi[k], 1.0-np.sum(local_phi[k])),
							   trace=False) for k in xrange(total_topics)]

	# Modeling the observations
	x = Container([Categorical('x_' + str(doc[0]), 
							   p=Lambda('phi_z_%i' % doc[0], 
							   			lambda z=z, local_phi=local_phi_lambda: [local_phi[z[n][order]]/np.sum(local_phi[z[n][order]]) for order,word in enumerate(doc[1])]), 
							   value=doc[1], 
							   observed=True) for n, doc in enumerate(local_docs)])

	return Model([beta, global_phi, local_phi, alpha, theta, z, x])

def step_function(mcmc):
	import pymc as pm 
	import numpy as np
	from numpy.random import normal as rnormal
	from numpy.random import poisson as rpoisson
	# from pymc.utils import round_array

	class CustomMetropolis(pm.Metropolis):
		'''
		Custom step method based on Metropolis, that enables to propose same values to global variables
		'''
		def __init__(self, stochastic, seed, scale=1., proposal_sd=None,
					 proposal_distribution=None, positive=False, verbose=-1, tally=True):

			# Initialize superclass
			pm.Metropolis.__init__(self,
								   stochastic,
								   scale=scale,
								   proposal_sd=proposal_sd,
								   proposal_distribution=proposal_distribution,
								   verbose=verbose,
								   tally=tally)

			current_state = np.random.get_state()
			np.random.seed(seed)
			self.random_state = np.random.get_state()
			np.random.set_state(current_state)

		def propose(self):
			current_state = np.random.get_state()
			np.random.set_state(self.random_state)
			if self.proposal_distribution == "Normal":
				self.stochastic.value = rnormal(
					self.stochastic.value,
					self.adaptive_scale_factor *
					self.proposal_sd,
					size=self.stochastic.value.shape)
			elif self.proposal_distribution == "Prior":
				self.stochastic.random()
			self.random_state = np.random.get_state()
			np.random.set_state(current_state)

	class HybridRandomWalk(pm.Metropolis):
		def __init__(self, stochastic, min_value, max_value, scale=1., proposal_sd=None,
					 proposal_distribution=None, positive=True, verbose=-1, tally=True):
			# HybridRandomWalk class initialization

			# Initialize superclass
			pm.Metropolis.__init__(self,
								   stochastic,
								   scale=scale,
								   proposal_sd=proposal_sd,
								   proposal_distribution=proposal_distribution,
								   verbose=verbose,
								   tally=tally)

			# Flag for positive-only values
			self._positive = positive

			self.min_value = min_value
			self.max_value = max_value

		_valid_proposals = ['RandomWalk']

		@staticmethod
		def competence(stochastic):
			"""
			The competence function for HybridRandomWalk.
			"""
			if stochastic.dtype in integer_dtypes:
				return 0.5
			else:
				return 0

		def propose(self):
			a = np.random.rand()
			k = np.shape(self.stochastic.value)
			'''if a < 0.3:
				# Propose new values using uniform distribution
				self.stochastic.value = np.random.randint(self.min_value, self.max_value, size=k)
			else:
				walk = np.random.randint(0, 2, size=k)
				zero_indices = walk == 0
				walk[zero_indices] = -1
				new_val = self.stochastic.value + walk
				low_values_indices = new_val < self.min_value
				new_val[low_values_indices] = self.min_value
				large_values_indices = new_val > self.max_value-1
				new_val[large_values_indices] = self.max_value-1
				# Round before setting proposed value
				self.stochastic.value = new_val'''
			# Random walk
			# Add or subtract (equal probability) Poisson sample
			new_val = self.stochastic.value + rpoisson(self.adaptive_scale_factor * self.proposal_sd) * (-np.ones(k)) ** (np.random.random(k) > 0.5)

			if self._positive:
				# Enforce positive values
				new_val = abs(new_val)
			large_values_indices = new_val > self.max_value-1
			new_val[large_values_indices] = 2*(self.max_value) - 1 - new_val[large_values_indices]
			self.stochastic.value = new_val

	import re
	# Apply this custom step method to all topic assignment variables, namely z's
	pattern = re.compile('z_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for z in params:
		mcmc.use_step_method(HybridRandomWalk, z, 0, total_topics)
	# Global parameters use the CustomMetropolis step method
	pattern = re.compile('global_phi_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for n, p in enumerate(params):
		mcmc.use_step_method(CustomMetropolis, p, phi_seeds[n])
	pattern = re.compile('beta_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for n, p in enumerate(params):
		mcmc.use_step_method(CustomMetropolis, p, beta_seeds[n])
	pattern = re.compile('global_phi_|local_phi_|theta_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for p in params:
		mcmc.use_step_method(pm.Metropolis, p, proposal_distribution="Prior")
	return mcmc

from pymc.DistributedMCMC import DistributedMCMC

# The path of the txt file that was produced by the preprocess_nips.py script
path = 'hdfs:///user/test/data/nips.txt'

m = DistributedMCMC(spark_context=sc, 
					model_function=model_function, 
					nJobs=4, 
					observation_file=path, 
					local_iter=1000, 
					step_function=step_function)

m.sample(1000)
