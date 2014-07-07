def model_function(data, global_param):
	'''
	data : list of lines where each line represents a document as follows: doc_number,word_0 word_1 ... word_N
	global_param : random number generator seed
	'''
	from pymc import Dirichlet, Categorical, Deterministic, Lambda, Stochastic
	import math
	import numpy as np
	from numpy.random import dirichlet

	beta = 0.1
	total_vocab = 78
	alpha = [beta for t in xrange(total_vocab)]
	total_topics = 10
	local_iter = 10

	rand_state = np.random.get_state()
	np.random.seed(global_param)
	topic_word_dist = [[dirichlet(alpha) for i in xrange(local_iter+1)] for t in xrange(total_topics)]
	np.random.set_state(rand_state)

	def log_beta(alpha):
		return sum(math.lgamma(a) for a in alpha) - math.lgamma(sum(alpha))
	    
	def phi_logp(value, alpha, topic):
		kernel = sum((a - 1) * math.log(t) for a, t in zip(alpha, value))
		return kernel - log_beta(alpha) 

	def phi_rand(alpha, topic):
		return topic_word_dist[topic].pop(0)

	phi = [Stochastic(logp=phi_logp,
					 doc='Dirichlet prior for topic-word distributions',
					 name='phi_%i' % k,
					 parents={'alpha': alpha,
					 		  'topic': k},
					 random=phi_rand,
					 trace=True,
					 dtype=float,
					 rseed=1.,
					 observed=False,
					 cache_depth=2,
					 plot=None,
					 verbose=0) for k in xrange(total_topics)]

	observations = list()
	local_docs = dict()
	doc_number = 0
	for line in data:
		document_data = line.split(',')
		local_docs[doc_number] = int(document_data[0])
		words = document_data[1].split(' ')
		for order, word in enumerate(words):
			observations.append((int(word), order, int(doc_number)))
		doc_number += 1

	alpha = 50/total_topics
	theta = [Dirichlet('theta_%i' % i, theta=[alpha for k in xrange(total_topics)]) for i in xrange(doc_number)]
	z = [Categorical('z_' + str(obs[1]) + '_' + str(local_docs[obs[2]]), p=theta[obs[2]], size=1) for obs in observations]
	x = list()
	for n, obs in enumerate(observations):
		p = Lambda('p_' + str(obs[1]) + '_' + str(local_docs[obs[2]]), lambda z=z[n]: phi[z].value)
		x.append(Categorical('x_' + str(obs[1]) + '_' + str(local_docs[obs[2]]), p=p, value=obs[0], size=1, observed=True))
	return locals()

def global_update():
	import random
	return int(random.random()*1000000)


from pymc.DistributedMCMC import DistributedMCMC

path = '/Users/test/Desktop/nips.txt'

m = DistributedMCMC(spark_context=sc, model_function=model_function, nJobs=4, observation_file=path, local_iter=10, global_update=('phi', global_update))

m.sample(40)
