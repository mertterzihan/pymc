total_partitions = 8
seed = 123456

def data_process(data):
	docs = list()
	# Given the data as a list of strings (lines), structure it in such a way that it can be used by the below model
	for line in data[1]:
		document_data = line.split(',')
		words = document_data[1].split(' ')
		words = map(int, words)
		docs.append((int(document_data[0]), words))
	return (data[0], docs)

def model_function(data, global_param):
	import pymc
	import numpy as np
	
	total_topics = 10
	vocab_length = 4792
	beta = 0.01
	alpha = 0.1

	current_state = np.random.get_state()
	np.random.seed(seed)
	initial_values = list()
	for doc in data[1]:
		doc_values = np.random.randint(total_topics, size=len(doc[1]))
		initial_values.append(doc_values)
	np.random.set_state(current_state)
	
	def logp(value, **kwargs):
		return 1

	if global_param is None:
		topic_word_counts = None
	else:
		topic_word_counts = global_param
		
	z = pymc.Stochastic(logp=logp,
						doc='',
						name='z_%i' % data[0],
						parents={'documents' : data[1],
								 'vocab_length' : vocab_length,
								 'alpha' : alpha,
								 'beta' : beta,
								 'total_topics' : total_topics,
								 'topic_word_counts' : topic_word_counts},
						value=initial_values,
						dtype=list)

	return pymc.Model([z])

def step_function(mcmc):
	import pymc
	import numpy as np 

	class CollapsedGibbs(pymc.Metropolis):
		def __init__(self, stochastic, scale=1., proposal_sd=None, proposal_distribution=None, 
					 positive=True, verbose=-1, tally=True):
			pymc.Metropolis.__init__(self,
								   stochastic,
								   scale=scale,
								   proposal_sd=proposal_sd,
								   proposal_distribution=proposal_distribution,
								   verbose=verbose,
								   tally=tally)

			self.alpha = self.stochastic.parents['alpha']
			self.beta = self.stochastic.parents['beta']
			self.vocab_length = self.stochastic.parents['vocab_length']
			self.docs = self.stochastic.parents['documents']
			self.total_topics = self.stochastic.parents['total_topics']
			self.topic_word_counts = self.stochastic.parents['topic_word_counts']

			if self.topic_word_counts is None:
				self.topic_word_counts = np.zeros((self.total_topics, self.vocab_length))
				self.topic_counts = np.zeros(self.total_topics)
				self.document_topic_counts = np.zeros((len(self.docs), self.total_topics))
				for doc_index, doc in enumerate(self.docs):
					for word_index, word in enumerate(doc[1]):
						topic_assignment = self.stochastic.value[doc_index][word_index]
						self.topic_counts[topic_assignment] += 1
						self.topic_word_counts[topic_assignment, word] += 1
						self.document_topic_counts[doc_index, topic_assignment] += 1

				self.topic_word_counts = np.add(self.topic_word_counts, self.beta)
				self.topic_counts = np.add(self.topic_counts, self.vocab_length * self.beta)
				self.document_topic_counts = np.add(self.document_topic_counts, self.alpha)
			else:
				self.topic_counts = self.topic_word_counts.sum(axis=1)
				self.document_topic_counts = np.zeros((len(self.docs), self.total_topics))
				for doc_index, doc in enumerate(self.docs):
					for word_index, word in enumerate(doc[1]):
						topic_assignment = self.stochastic.value[doc_index][word_index]
						self.document_topic_counts[doc_index, topic_assignment] += 1

				self.document_topic_counts = np.add(self.document_topic_counts, self.alpha)
				self.topic_word_counts = np.add(self.topic_word_counts, self.beta)
				self.topic_counts = np.add(self.topic_counts, self.vocab_length * self.beta)
			self.old_topic_word_counts = self.topic_word_counts

		def step(self):
			new_assignments = list()
			for doc_index, doc in enumerate(self.docs):
				doc_topic_assignments = np.zeros(len(doc[1]))
				for word_index, word in enumerate(doc[1]):
					prev_assignment = self.stochastic.value[doc_index][word_index]
					if self.topic_word_counts[prev_assignment, word] < 1:
						neg = True
					else:
						self.topic_counts[prev_assignment] -= 1
						self.topic_word_counts[prev_assignment, word] -= 1
						neg = False
					self.document_topic_counts[doc_index, prev_assignment] -= 1

					mult_probabilities = np.divide(np.multiply(self.topic_word_counts[:, word], self.document_topic_counts[doc_index, :]), self.topic_counts)
					mult_probabilities = np.divide(mult_probabilities, mult_probabilities.sum())
					topic_assignment = np.random.multinomial(1, mult_probabilities).argmax()

					if not neg:
						self.topic_counts[topic_assignment] += 1
						self.topic_word_counts[topic_assignment, word] += 1
					else:
						indices = self.topic_word_counts[:,word] < 1.0/self.total_topics + self.beta
						if indices.any():
							tmp = np.subtract(self.topic_word_counts[:,word], 1.0/self.total_topics+self.beta)
							total = np.sum(tmp[indices])
							tmp[np.invert(indices)] -= (total/tmp[np.invert(indices)].shape[0])
							tmp[indices] = 0
							self.topic_word_counts[:,word] = np.add(tmp, self.beta) 
						else:
							self.topic_word_counts[:, word] -= 1.0/self.total_topics
						self.topic_word_counts[topic_assignment, word] += 1
						self.topic_counts = self.topic_word_counts.sum(axis=1)
					self.document_topic_counts[doc_index, topic_assignment] += 1
					doc_topic_assignments[word_index] = topic_assignment
				new_assignments.append(doc_topic_assignments)
			self.stochastic.value = new_assignments

	import re
	pattern = re.compile('z_')
	params = [p for p in mcmc.variables if pattern.match(p.__name__)]
	for z in params:
		mcmc.use_step_method(CollapsedGibbs, z)
	return mcmc

def global_update(rdd):
	from numpy import add
	import numpy as np
	result = rdd.map(lambda x: x[3]).reduce(add)
	for column in xrange(result.shape[1]):
		pos_indices = result[:,column] > 0
		if pos_indices.all():
			total = np.sum(result[np.invert(pos_indices),column])
			result[pos_indices,column] -= (total/result[pos_indices,column].shape[0])
			result[np.invert(pos_indices),column] = 0
	return result

def sample_return(mcmc):
	import re
	import numpy as np
	pattern = re.compile('z_')
	z = [p for p in mcmc.step_method_dict.keys() if pattern.match(p.__name__)]
	step_method = mcmc.step_method_dict[z[0]][0]
	beta = 0.01
	topic_word_counts = np.subtract(step_method.topic_word_counts, beta)
	old_topic_word_counts = np.subtract(step_method.old_topic_word_counts, beta)
	return np.subtract(topic_word_counts, np.multiply(float(total_partitions-1)/total_partitions, old_topic_word_counts))


from pymc.DistributedMCMC import DistributedMCMC

# The path of the txt file that was produced by the preprocess_nips.py script
#path = '/user/mert.terzihan/data/nips.txt'
#path = '/home/mert.terzihan/tmp/nips.txt'
path = '/Users/mert.terzihan/Desktop/txt/nips.txt'

#sc.addPyFile('/home/mert.terzihan/pymc/pymc/dist/pymc-2.3.4-py2.6-linux-x86_64.egg')

m = DistributedMCMC(spark_context=sc, 
					model_function=model_function,
					data_process=data_process, 
					nJobs=total_partitions, 
					observation_file=path, 
					local_iter=1, 
					step_function=step_function,
					global_update=global_update,
					sample_return=sample_return)

m.sample(3)