def model_function(data, phi):
	'''
	data : list of tuples (word, order_in_doc, doc)
	local_docs : list of docs (ints)
	'''
	from pymc import Dirichlet, Categorical, Deterministic, Lambda

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

	total_topics = len(phi)
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
	total_vocab = 4844
	total_topics = 10
	beta_vector = [beta for t in xrange(total_vocab)]
	return [dirichlet(beta_vector) for k in xrange(total_topics)]

from pymc.DistributedMCMC import DistributedMCMC

path = '/Users/test/Documents/nips.txt'

m = DistributedMCMC(spark_context=sc, model_function=model_function, nJobs=100, observation_file=path, local_iter=100, global_update=('phi', global_update))

m.sample(500)
