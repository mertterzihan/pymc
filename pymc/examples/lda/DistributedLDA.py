def model_function(data, global_param):
	'''
	data : list of tuples (word, order_in_doc, doc)
	global_param : a list of local_iter elements where each element contains 
				   topic-word distribution for each iteration
	'''
	from pymc import Dirichlet, Categorical, Deterministic, Lambda, Stochastic
	import math

	total_topics = len(global_param[0])
	topic_word_dist = global_param

	beta = 0.1
	alpha = [beta for t in xrange(total_vocab)]

	def log_beta(alpha):
    	return sum(math.lgamma(a) for a in alpha) - math.lgamma(sum(alpha))
	    
	def phi_logp(value):
		kernel = sum((a - 1) * math.log(t) for a, t in zip(alpha, value))
	    return kernel - log_beta(alpha) 

	def phi_rand():
		return topic_word_dist.popleft()

	phi = Stochastic(logp=phi_logp,
					 doc='Dirichlet prior for topic-word distributions',
					 name='phi',
					 parents={},
					 random=phi_rand,
					 trace=True,
					 dtype=list,
					 rseed=1.,
					 observed=False,
					 cache_depth=2,
					 plot=None,
					 verbose=0)

	observations = list()
	local_docs = dict()
	#lines = data.split('\n')
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
		p = Lambda('p_' + str(obs[1]) + '_' + str(local_docs[obs[2]]), lambda z=z[n]: phi[z])
		x.append(Categorical('x_' + str(obs[1]) + '_' + str(local_docs[obs[2]]), p=p, value=obs[0], size=1, observed=True))
	return locals()

def global_update():
	from numpy.random import dirichlet
	beta = 0.1
	total_vocab = 78
	total_topics = 10
	local_iter = 10
	beta_vector = [beta for t in xrange(total_vocab)]
	return [[dirichlet(beta_vector) for k in xrange(total_topics)] for i in local_iter]

from pymc.DistributedMCMC import DistributedMCMC

path = '/Users/mert.terzihan/Desktop/nips.txt'

m = DistributedMCMC(spark_context=sc, model_function=model_function, nJobs=4, observation_file=path, local_iter=10, global_update=('phi', global_update))

m.sample(40)
