'''
Python module for distributing an MCMC model among Spark clusters. 

It distributes the observation data using Spark's built-in textFile function.
'''

__all__ = ['DistributedMCMC']

from .MCMCSpark import MCMCSpark
from .MCMC import MCMC
from pymc.database import distributed_spark

class DistributedMCMC(MCMCSpark):

	def __init__(self, input=None, db='spark', name='MCMC', calc_deviance=True, nJobs=1, **kwargs):
		'''
		Parameters
		----------
		- input : model
			Model definition
		- db : str
			The name of the database backend that will store the values
			of the stochastics and deterministics sampled during the MCMC loop.
		- nJobs : integer
			Number of Spark jobs that will run MCMC
		- **kwarg : dict
			- spark_context : SparkContext
				A SparkContext instance that will be used to load data
			- dbname : str
				Optional, location to save the files on HDFS
			- model_function : function
				A wrapper function which builds the model and returns it
			- observation_file : HDFS path
				Path of the data file which will be partitioned over the machines in the cluster
			- local_iter : int
				Number of iterations that local sampler will be run
			- global_update : function
				A wrapper function which updates the global parameters
		'''
		self.model_function = kwargs.pop("model_function", None)
		self.observation_file = kwargs.pop("observation_file", None)
		self.local_iter = kwargs.pop("local_iter", None)
		self.global_update = kwargs.pop("global_update", None)
		MCMCSpark.__init__(self, input=None, db=db, name=name, calc_deviance=calc_deviance, nJobs=nJobs, **kwargs)

	def sample(
		self, iter, burn=0, thin=1, tune_interval=1000, tune_throughout=True,
		save_interval=None, burn_till_tuned=False, stop_tuning_after=5,
			verbose=0, progress_bar=True):
		'''
		Partitions the data over the machines, runs sampling algorithm on the slaves and 
		updates global parameter in turn until reaching the specified number of iterations

		Parameters are similar to MCMC.sample()
		'''
		name = self.name
		calc_deviance = self.calc_deviance
		kwargs = self.kwargs
		model_function = self.model_function
		observation_file = self.observation_file
		local_iter = self.local_iter
		nJobs = self.nJobs
		global_update = self.global_update

		def sample_on_spark(data):
			# Load the database, so that MCMC can continue sampling from where it had concluded the previous local iteration
			def load_ram_database(data_dict):
				from pymc.database import ram
				db = ram.Database('temp_database')
				trace_names = list()
				for key in data_dict.keys():
					if key != '_state_':
						trace_names.append(key)
						db._traces[key] = ram.Trace(name=key, value={0:data_dict[key][-2:-1]}, db=db)
						setattr(db, key, db._traces[key])
					else:
						db._state_ = data_dict[key]
				db.trace_names.append(trace_names)
				return db

			if len(data) == 3: # If this method has been run in a previous iteration, so that MCMC should be loaded
				input_model = model_function(data[1], global_param.value)
				index = data[2].index(None)
				m = MCMC(input_model, db=load_ram_database(data[2][index-1]), name=name, calc_deviance=calc_deviance, **kwargs)
			else: # If this is the first iteration
				input_model = model_function(data[1], global_param.value)
				m = MCMC(input_model, db='ram', name=name, calc_deviance=calc_deviance, **kwargs)

			m.sample(local_iter, burn, thin, tune_interval, tune_throughout,
        		save_interval, burn_till_tuned, stop_tuning_after,
            	verbose, progress_bar)

			# TODO: Local Update

			# Create or update the dictionary
			if len(data) == 3:
				import numpy as np
				container_list = data[2]
				index = container_list.index(None)
				container = {}
				for tname in m.db._traces:
					container[tname] = m.trace(tname)[:]
				container['_state_'] = m.get_state()
				container_list[index] = container
				return (data[0], data[1], container_list)
			else:
				container_list = [None]*(iter/local_iter)
				container = {}
				for tname in m.db._traces:
					container[tname] = m.db._traces[tname]._trace[0]
				container['_state_'] = m.get_state()
				container_list[0] = container
				return (data[0], data[1], container_list)

		def generate_keys(splitIndex, iterator):
			for i in iterator:
				yield (splitIndex,i)

		def generate_lists(a, b):
			if isinstance(a, list):
				if isinstance(b, list):
					return a + b
				else:
					a.append(b)
					return a
			elif isinstance(b, list):
				b.append(a)
				return b
			else:
				return list([a, b])
		# Partition the data and generate a list of data assigned to each node
		rdd = self.sc.textFile(observation_file, minPartitions=nJobs).mapPartitionsWithIndex(generate_keys).reduceByKey(generate_lists).cache()
		current_iter = 0
		while current_iter < iter:
			# If the user has provided a global update function, execute it to synch the nodes
			if self.global_update is not None:
				param = global_update[1]()
				global_param = self.sc.broadcast(param) # Broadcast the global parameters
			rdd = rdd.map(sample_on_spark).cache() # Run the local sampler
			current_iter += self.local_iter
		rdd = rdd.map(lambda x: (x[0], x[2])).cache()
		def extract_var_names(a,b):
			if isinstance(a, set):
				if isinstance(b, set):
					a.update(b)
				else:
					a.add(b)
				return a
			elif isinstance(b, set):
				b.add(a)
				return b
			else:
				s = set([a,b])
				return s
		# Extract the variable names
		vars_to_tally = rdd.map(lambda x: x[1][0]).flatMap(lambda x: filter(lambda i: i!='_state_', x.keys())).reduce(extract_var_names)
		self._variables_to_tally = vars_to_tally
		self._assign_database_backend(rdd, vars_to_tally)
		# If hdfs was selected as the database, save the traces as txt files
		if self.save_to_hdfs:
			self.save_as_txt_file(self.dbname)


	def _assign_database_backend(self, db, vars_to_tally):
		'''
		Assign distributed_spark RDD database
		'''
		self.db = distributed_spark.Database(db, vars_to_tally)

	def save_as_txt_file(self, path, chain=None):
		'''
		Save the data to HDFS as txt files

		Parameters
		----------
		path : str
			Name of the file to save the data
		chain : int 
			The index of the chain to be saved. Defaults to None (all chains)
		'''
		temp_rdd = self.db.rdd
		if chain is not None:
			if chain < 0:
				chain = xrange(self.db.chains)[chain]
			self.save_txt_helper(path, chain)
		else:
			total_chains = temp_rdd.map(lambda x: len(x[1])).first()
			for chain in xrange(total_chains):
				self.save_txt_helper(path, chain)
			

	def save_txt_helper(self, path, chain):
		'''
		Helper function for saving data to HDFS as txt files
		'''
		import datetime
		import os
		import numpy as np
		from numpy.compat import asstr
		for var in self._variables_to_tally:
			def save_mapper(x):
				data = '# Variable: %s\n' % var
				data += '# Partition: %s\n' % x[0]
				data += '# Sample shape: %s\n' % str(x[1].shape)
				data += '# Date: %s\n' % datetime.datetime.now()
				X = x[1].reshape((-1, x[1][0].size))
				fmt = '%.18e'
				delimiter = ','
				newline = '\n'
				if isinstance(fmt, bytes):
					fmt = asstr(fmt)
				delimiter = asstr(delimiter)
				X = np.asarray(X)
				if X.ndim == 1:
					if X.dtype.names is None:
						X = np.atleast_2d(X).T
						ncol = 1
					else:
						ncol = len(X.dtype.descr)
				else:
					ncol = X.shape[1]
				n_fmt_chars = fmt.count('%')
				fmt = [fmt, ] * ncol
				format = delimiter.join(fmt)
				for row in X:
					data += format % tuple(row) + newline
				return data

			self.db.rdd.filter(lambda x: var in x[1][chain]).map(lambda x: (x[0], x[1][chain][var])).map(save_mapper).saveAsTextFile(os.path.join(path, str(chain), var))
		self.db.rdd.map(lambda x: (x[0], x[1][chain]['_state_'])).saveAsTextFile(os.path.join(path, str(chain), 'state'))
