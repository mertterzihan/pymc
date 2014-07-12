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
	from pymc import Dirichlet, Categorical, Deterministic, Lambda, Stochastic
	import math
	import numpy as np
	from numpy.random import dirichlet

	beta = 0.1
	total_vocab = 78
	beta_vector = [beta for t in xrange(total_vocab)]
	total_topics = 10
	local_iter = 10

	'''
	Save the state of numpy random number generator, generate topic-word samples from a symmetric Dirichlet
	distribution defined by global_param (the same phi vectors will be drawn in each machine in the cluster).
	Then revert the state of numpy random number generator back to the initial state. 
	'''
	rand_state = np.random.get_state()
	np.random.seed(global_param)
	# Symmetric Dirichlet distribution for topic-word distribution, phi, defined by beta
	topic_word_dist = [[dirichlet(beta_vector) for i in xrange(local_iter+1)] for t in xrange(total_topics)]
	np.random.set_state(rand_state)

	def log_beta_func(beta_vector):
		'''
		A function to compute log-beta function given the prior parameters of a Dirichlet distribution
		'''
		return sum(math.lgamma(a) for a in beta_vector) - math.lgamma(sum(beta_vector))

	# Precalculate log_beta, since they share the same prior parameter, it is the same for all the cases
	log_beta = log_beta_func(beta_vector)
	    
	def phi_logp(value, beta_vector, topic, log_beta):
		'''
		Compute the pdf of a value given the Dirichlet parameters
		'''
		kernel = sum((a - 1) * math.log(t) for a, t in zip(beta_vector, value))
		return kernel - log_beta

	def phi_rand(beta_vector, topic, log_beta):
		'''
		Since the random variables have been already drawn, it just extracts the next 
		'''
		return topic_word_dist[topic].pop(0)

	# Custom Stochastic class for the node that represents topic-word distribution 
	phi = [Stochastic(logp=phi_logp,
					 doc='Dirichlet prior for topic-word distributions',
					 name='phi_%i' % k,
					 parents={'beta_vector': beta_vector,
					 		  'topic': k,
					 		  'log_beta': log_beta},
					 random=phi_rand,
					 trace=True,
					 dtype=float,
					 rseed=1.,
					 observed=False,
					 cache_depth=2,
					 plot=None,
					 verbose=0) for k in xrange(total_topics)]

	local_docs = list()
	# Given the data as a list of strings (lines), structure it in such a way that it can be used by the below model
	for doc_number,line in enumerate(data):
		document_data = line.split(',')
		words = document_data[1].split(' ')
		words = map(int, words)
		local_docs.append((int(document_data[0]), words))

	# The symmetric prior parameter for document-topic distribution
	alpha = 50/total_topics
	# The Dirichlet distribution for document-topic distribution, theta
	theta = [Dirichlet('theta_%i' % local_docs[i][0], theta=[alpha for k in xrange(total_topics)]) for i in xrange(len(local_docs))]
	# The topic assignments for each word
	z = [Categorical('z_' + str(doc[0]), p=theta[n], size=len(doc[1])) for n, doc in enumerate(local_docs)]
	# Modeling the observations
	x = [Categorical('x_' + str(doc[0]), p=[phi[z[n][order].value].value for order,word in enumerate(doc[1])], value=doc[1], size=len(doc[1]), observed=True) for n, doc in enumerate(local_docs)]
	
	return locals()

def global_update():
	'''
	This function is being called after each local iteration of sampler to synchronize the nodes
	'''
	import random
	return int(random.random()*1000000)


from pymc.DistributedMCMC import DistributedMCMC

# The path of the txt file that was produced by the preprocess_nips.py script
path = '/Users/test/nips.txt'

m = DistributedMCMC(spark_context=sc, model_function=model_function, nJobs=4, observation_file=path, local_iter=10, global_update=('phi', global_update))

m.sample(40)
