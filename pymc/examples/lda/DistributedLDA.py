from numpy.random import randint
total_topics = 10
phi_seeds = randint(1000000, size=total_topics)

def model_function(data, global_param):
	'''
	This is the function which generates the PyMC model for LDA (naively distributed)

	Parameters
	----------
	data : list of strings
		list of lines where each line represents a document as follows: doc_number,word_0 word_1 ... word_N
	global_param : int
		random number generator seed
	'''
	from pymc import Dirichlet, Categorical, Gamma, Lambda, Container, Model
	import numpy as np

	beta = 0.01
	total_vocab = 4792 #78 #614
	beta_vector = [beta for t in xrange(total_vocab)]

	phi_value = [1.0/total_vocab for t in xrange(total_vocab)]

	# Symmetric Dirichlet prior for topic-word distributions
	phi = Container([Dirichlet("phi_%s" % k, 
							   theta=beta_vector,
							   value=phi_value[:-1]) for k in range(total_topics)])

	local_docs = list()
	# Given the data as a list of strings (lines), structure it in such a way that it can be used by the below model
	for doc_number,line in enumerate(data):
		document_data = line.split(',')
		words = document_data[1].split(' ')
		words = map(int, words)
		local_docs.append((int(document_data[0]), words))

	# The symmetric prior parameter for document-topic distribution
	alpha = 0.1#50.0/total_topics
	alpha_vector = [alpha for k in xrange(total_topics)]
	theta_value = [1.0/total_topics for i in xrange(total_topics)]
	# The Dirichlet distribution for document-topic distribution, theta
	theta = Container([Dirichlet('theta_%i' % local_docs[i][0], 
								 theta=alpha_vector,
								 value=theta_value[:-1]) for i in xrange(len(local_docs))])

	# The topic assignments for each word
	z = Container([Categorical('z_' + str(doc[0]), 
							   p=theta[n], 
							   size=len(doc[1]), 
							   verbose=0) for n, doc in enumerate(local_docs)])

	phi_lambda = [Lambda('phi_lambda_%i' % k,
							   lambda phi=phi: np.append(phi[k], 1.0-np.sum(phi[k])),
							   trace=False) for k in xrange(total_topics)]

	# Modeling the observations
	x = Container([Categorical('x_' + str(doc[0]), 
							   p=Lambda('phi_z_%i' % doc[0], 
							   			lambda z=z, phi_lambda=phi_lambda: [phi_lambda[z[n][order]] for order,word in enumerate(doc[1])]), 
							   value=doc[1], 
							   observed=True) for n, doc in enumerate(local_docs)])

	return Model([theta, phi, z, x])

# Define a custom step to prevent PyMC to get stuck and reject proposals
def step_function(mcmc):
	import pymc
	import numpy as np
	from numpy.random import normal as rnormal
	from numpy.random import poisson as rpoisson

	class CustomMetropolis(pymc.Metropolis):
		'''
		Custom step method based on Metropolis, that enables to propose same values to global variables
		'''
		def __init__(self, stochastic, seed, scale=1., proposal_sd=None,
					 proposal_distribution=None, positive=True, verbose=-1, tally=True):

			# Initialize superclass
			pymc.Metropolis.__init__(self,
								   stochastic,
								   scale=scale,
								   proposal_sd=proposal_sd,
								   proposal_distribution=proposal_distribution,
								   verbose=verbose,
								   tally=tally)

			self._positive = positive
			current_state = np.random.get_state()
			np.random.seed(seed)
			self.random_state = np.random.get_state()
			np.random.set_state(current_state)

		def propose(self):
			current_state = np.random.get_state()
			np.random.set_state(self.random_state)
			if self.proposal_distribution == "Normal":
				new_val = rnormal(
					self.stochastic.value,
					self.adaptive_scale_factor *
					self.proposal_sd,
					size=self.stochastic.value.shape)
				if self._positive:
					# Enforce positive values
					new_val = abs(new_val)
				new_val = new_val/np.sum(new_val)
				self.stochastic.value = new_val
			elif self.proposal_distribution == "Prior":
				self.stochastic.random()
			self.random_state = np.random.get_state()
			np.random.set_state(current_state)


	class HybridRandomWalk(pymc.Metropolis):
		def __init__(self, stochastic, min_value, max_value, scale=1., proposal_sd=None,
					 proposal_distribution=None, positive=True, verbose=-1, tally=True):
			# HybridRandomWalk class initialization

			# Initialize superclass
			pymc.Metropolis.__init__(self,
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
	pattern = re.compile('phi_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for n, p in enumerate(params):
		mcmc.use_step_method(CustomMetropolis, p, phi_seeds[n])
	pattern = re.compile('theta_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for p in params:
		mcmc.use_step_method(pymc.Metropolis, p, proposal_distribution="Prior")
	return mcmc


from pymc.DistributedMCMC import DistributedMCMC

# The path of the txt file that was produced by the preprocess_nips.py script
path = '/user/mert.terzihan/data/nips.txt'
#path = '/home/mert.terzihan/tmp/nips.txt'
#path = '/Users/mert.terzihan/Desktop/nips.txt'

sc.addPyFile('/home/mert.terzihan/pymc/pymc/dist/pymc-2.3.4-py2.6-linux-x86_64.egg')

m = DistributedMCMC(spark_context=sc, 
					model_function=model_function, 
					nJobs=72, 
					observation_file=path, 
					local_iter=10, 
					step_function=step_function)

m.sample(10)
