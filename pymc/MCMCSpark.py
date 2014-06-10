'''
Python module for running MCMC on Spark clusters

Currently, it only supports HDFS as backend

For an example usage, please look at pymc/examples/disaster_model_spark.py
'''

__all__ = ['MCMCSpark']

from .MCMCMultiChain import MCMCMultiChain
from pymc.database import ram

import numpy as np 
import os
from pyspark import SparkContext
from pymc import six

class MCMCSpark():
	'''
	For each Spark job, an MCMC instance is created
	status.txt belongs to the last job
	'''
	def __init__(self, input=None, db='hdfs', name='MCMC', calc_deviance=True, nJobs=1, **kwargs):
		'''
		Parameters
		----------
		- input : model
			Model definition
		- db : str
			The name of the database backend that will store the values
			of the stochastics and deterministics sampled during the MCMC loop.
		- nJobs : integer
			Number of threads that will run MCMC
		- **kwarg : dict
			- spark_host : str
				Cluster URL to connect to
			- spark_home : str
				Location where Spark is installed on cluster nodes
			- model_file : str
				Location of the file that defines model to be loaded
			- obs_files : str
				Location of the files that defines observations. By default it is None.
			- dbname : str
				Location on HDFS that the traces will be saved without a leading '/'
			- hdfs_host : str
				The IP address or host name of the HDFS namenode
			- port : str
				The port number for WebHDFS on namenode
			- user_name : str
				WebHDFS user name used for authentication
		'''
		if nJobs < 1:
			nJobs = 1
		self.spark_host = kwargs.pop("spark_host", None)
		self.spark_home = kwargs.pop("spark_home", None)
		self.model_file = kwargs.pop("model_file", None)
		if not all([self.spark_host, self.spark_home, self.model_file]):
			raise ValueError('Spark Host and Spark Home cannot be None!')
		self.obs_files = kwargs.pop("obs_files", None)
		self.nJobs = nJobs
		self.db = db
		self.name = name
		self.calc_deviance = calc_deviance
		self.kwargs = kwargs
		self.sc = SparkContext(master=self.spark_host, appName=self.name+self.db, sparkHome=self.spark_home)
		if self.obs_files is not None:
			for f in self.obs_files:
				self.sc.addFile(f)
		self.sc.addPyFile(self.model_file)

	def sample(
		self, iter, burn=0, thin=1, tune_interval=1000, tune_throughout=True,
		save_interval=None, burn_till_tuned=False, stop_tuning_after=5,
			verbose=0, progress_bar=True):
		'''
		Starts Spark jobs that initialize traces, run sampling loop, clean up afterwards
		Similar to MCMC.sample() in terms of parameters
		Returns a RAM database to the user
		'''
		db = self.db
		name = self.name
		calc_deviance = self.calc_deviance
		kwargs = self.kwargs
		model_file = self.model_file
		total_jobs = self.nJobs

		def sample_on_spark(nJob):
			model_module = os.path.splitext(os.path.basename(model_file))[0]
			imported_module = __import__(model_module)
			final_job = False
			if total_jobs-1 == nJob:
				final_job = True
			m = MCMCMultiChain(imported_module, db=db, name=name, calc_deviance=calc_deviance, chainNum=nJob, final_job=final_job, **kwargs)
			m.sample(iter, burn, thin, tune_interval, tune_throughout,
        		save_interval, burn_till_tuned, stop_tuning_after,
            	verbose, progress_bar)
			container = {}
			for tname in m.db._traces:
				container[tname] = m.db._traces[tname]._trace
			container['_state_'] = m.get_state()
			return container

		container = self.sc.parallelize(xrange(self.nJobs)).map(sample_on_spark).collect()
		data = {}
		for chain, traces in enumerate(container):
			for name in traces.keys():
				if chain == 0:
					data[name] = {}
				if name != '_state_':
					data[name][chain] = traces[name][0]

		db = ram.Database('MCMCSparkRamDatabase')
		db.chains = self.nJobs
		for name, values in six.iteritems(data):
			db._traces[name] = ram.Trace(name=name, value=values, db=db)
			setattr(db, name, db._traces[name])
		db._state_ = container[self.nJobs-1]['_state_']

		return db


def load_data(file_name, shape=None, delimiter=','):
	'''
	If any observation file has been provided in __init__, then they can be loaded by this method

	Parameters
	__________
	- file_name : str
		Name of the file to be loaded. It needs to reside on the Spark cluster
	- shape : list
		Shape of the matrix that is located in the file to be loaded
	-delimiter : str
		Character that splits elements in the file
	'''
	arr = np.loadtxt(file_name, delimiter=delimiter)
	if shape is not None:
		arr = arr.reshape(shape)
	return arr
	