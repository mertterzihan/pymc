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
	from pymc import Dirichlet, Categorical, Deterministic, Lambda, Stochastic, Container, Model
	import math
	import numpy as np
	from numpy.random import dirichlet
	from pymc.distributions import dirichlet_like

	beta = 0.5
	total_vocab = 78
	beta_vector = [beta for t in xrange(total_vocab+1)]
	total_topics = 10

	# Symmetric Dirichlet prior for topic-word distributions
	phi = Container([Dirichlet("phi_%s" % k, 
							   theta=beta_vector) for k in range(total_topics)])

	local_docs = list()
	# Given the data as a list of strings (lines), structure it in such a way that it can be used by the below model
	for doc_number,line in enumerate(data):
		document_data = line.split(',')
		words = document_data[1].split(' ')
		words = map(int, words)
		local_docs.append((int(document_data[0]), words))

	# The symmetric prior parameter for document-topic distribution
	alpha = 50.0/total_topics
	# The Dirichlet distribution for document-topic distribution, theta
	theta = Container([Dirichlet('theta_%i' % local_docs[i][0], 
								 theta=[alpha for k in xrange(total_topics)]) for i in xrange(len(local_docs))])
	# The topic assignments for each word
	z = Container([Categorical('z_' + str(doc[0]), 
							   p=theta[n], 
							   size=len(doc[1]), 
							   verbose=0) for n, doc in enumerate(local_docs)])
	# Modeling the observations
	x = Container([Categorical('x_' + str(doc[0]), 
							   p=Lambda('phi_z_%i' % doc[0], 
							   			lambda z=z, 
							   			phi=phi: [phi[z[n][order]]/np.sum(phi[z[n][order]]) for order,word in enumerate(doc[1])]), 
							   value=doc[1], 
							   observed=True) for n, doc in enumerate(local_docs)])
	return Model([theta, phi, z, x])

# Define a custom step to prevent PyMC to get stuck and reject proposals
def step_function(mcmc):
	import pymc as pm
	import numpy as np
	class UniformWalkStep(pm.Metropolis):
		def __init__(self, stochastic, min_value, max_value, scale=1., proposal_sd=None,
					 proposal_distribution=None, positive=False, verbose=-1, tally=True):
			# UniformWalkStep class initialization

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
			The competence function for UniformWalkStep.
			"""
			if stochastic.dtype in integer_dtypes:
				return 0.5
			else:
				return 0

		def propose(self):
			# Propose new values using uniform distribution
			k = np.shape(self.stochastic.value)
			self.stochastic.value = np.random.randint(self.min_value, self.max_value, size=k)

	K = 5
	import re
	# Apply this custom step method to all topic assignment variables, namely z's
	pattern = re.compile('z_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for z in params:
		mcmc.use_step_method(UniformWalkStep, z, 0, K)
	return mcmc


from pymc.DistributedMCMC import DistributedMCMC

# The path of the txt file that was produced by the preprocess_nips.py script
path = '/Users/test/nips.txt'
#global_update=('phi', global_update)
m = DistributedMCMC(spark_context=sc, 
					model_function=model_function, 
					nJobs=4, 
					observation_file=path, 
					local_iter=10, 
					step_function=step_function)

m.sample(100)
