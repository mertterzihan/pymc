'''
Python module for running MCMC on Spark clusters

Currently, it only supports HDFS as backend

For an example usage, please look at pymc/examples/disaster_model_spark.py
'''

__all__ = ['MCMCSpark']

from .MCMC import MCMC
from pymc.database import spark
import re


import numpy as np 
import os
from pyspark import SparkContext
from pymc import six
import copy
import itertools
import datetime
import os
from numpy.compat import asbytes, asstr
print_ = six.print_

Supported_Backends = ['spark', 'hdfs']

class MCMCSpark():
	'''
	For each Spark job, an MCMC instance is created
	status.txt belongs to the last job
	'''
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
		self._check_database_backend(db)
		self.save_to_hdfs = False
		if db == 'hdfs':
			self.save_to_hdfs = True
		if self.save_to_hdfs:
			self.dbname = kwargs.pop("dbname", None)
			if self.dbname is None:
				raise ValueError('Please provide a directory on HDFS to save files')
		self.sc = kwargs.pop("spark_context", None)
		if self.sc is None:
			raise ValueError('Please provide SparkContext')
		# self.model_file = kwargs.pop("model_file", None)
		self.input = input
		# if self.model_file is None and self.input is None:
			# raise ValueError('Please provide a model')
		self.nJobs = nJobs
		self.name = name
		self.calc_deviance = calc_deviance
		# self.sc.addPyFile(self.model_file)
		self.kwargs = kwargs

	def sample(
		self, iter, burn=0, thin=1, tune_interval=1000, tune_throughout=True,
		save_interval=None, burn_till_tuned=False, stop_tuning_after=5,
			verbose=0, progress_bar=True):
		'''
		Starts Spark jobs that initialize traces, run sampling loop, clean up afterwards
		Similar to MCMC.sample() in terms of parameters
		Returns a RAM database to the user
		'''
		name = self.name
		calc_deviance = self.calc_deviance
		kwargs = self.kwargs
		#model_file = self.model_file
		total_jobs = self.nJobs
		input_model = self.input

		def sample_on_spark(nJob):
			# model_module = os.path.splitext(os.path.basename(model_file))[0]
			# imported_module = __import__(model_module)
			final_job = False
			if total_jobs-1 == nJob:
				final_job = True
			m = MCMC(input_model, db='ram', name=name, calc_deviance=calc_deviance, **kwargs)
			m.sample(iter, burn, thin, tune_interval, tune_throughout,
        		save_interval, burn_till_tuned, stop_tuning_after,
            	verbose, progress_bar)
			container = {}
			for tname in m.db._traces:
				container[tname] = m.db._traces[tname]._trace[0]
			container['_state_'] = m.get_state()
			return (nJob, container)

		rdd = self.sc.parallelize(xrange(self.nJobs)).map(sample_on_spark).cache()
		#vars_to_tally = copy.copy(rdd.map(lambda x: x[1].keys())).first()
		vars_to_tally = rdd.map(lambda x: x[1].keys()).first()
		vars_to_tally.remove('_state_')
		self._variables_to_tally = set(vars_to_tally)
		self._assign_database_backend(rdd, vars_to_tally)
		if self.save_to_hdfs:
			self.save_as_pickle_object(self.dbname)

	def _check_database_backend(self, db):
		'''
		Check that given database is compatible with spark
		'''
		if not isinstance(db, str):
			raise ValueError('MCMCSpark supports just strings as parameter for database backend for now')
		elif db not in Supported_Backends:
			raise ValueError('MCMCSpark supports currently Spark (db=\'spark\') and HDFS (db=\'hdfs\') as database backends')
		self.save_to_hdfs = False
		if db == 'hdfs':
			self.save_to_hdfs = True

	def _assign_database_backend(self, db, vars_to_tally):
		'''
		Assign Spark RDD database
		'''
		self.db = spark.Database(db, vars_to_tally)

	def trace(self, name, chain=-1):
		'''
		Return the trace of a tallyable object stored in RDD

		Parameters
		----------
		name : str
			The name of the tallyable object
		chain : int
			The trace index. Setting 'chain=i' will return the trace created by the ith spark job
		'''
		if isinstance(name, str):
			return self.db.trace(name, chain)
		elif isinstance(name, Variable):
			return self.db.trace(name.__name__, chain)
		else:
			raise ValueError(
				'Name argument must be string or Variable, got %s.' %
				name)

	def get_state(self):
		'''
		Return the sampler's current state
		'''
		return self.db.getstate()

	def summary(self, variables=None, alpha=0.05, start=0, batches=100,
				chain=None, roundto=3):
		'''
		Generate a pretty-printed summary of the node.

		Parameters
		----------
		variables : list of str
			List of variable names to be summarized. By default it summarizes every variable
		alpha : float
			The alpha level for generating posterior intervals. Defaults to 0.05
		start : int
			The starting index from which to summarize chain. Defaults to zero
		batches : int 
			Batch size for calculating standard deviation for non-independent samples.
			Defaults to 100
		chain : int
			The index for which chain to summarize. Defaults to None (all chains)
		roundto : int
			The number of digits to round posterior statistics.
		'''
		pattern = re.compile(".*adaptive_scale_factor|deviance")
		if variables is None:
			variables = [v for v in self._variables_to_tally if not pattern.match(v)]
		else:
			variables = [v for v in variables if (v in self._variables_to_tally) and (not pattern.match(v))]
		for variable in variables:
			statdict = self.db._traces[variable].stats(alpha=alpha, start=start, 
													   batches=batches, chain=chain)
			self.summary_helper(variable, statdict, roundto)

	def summary_helper(self, variable, statdict, roundto=3):
		'''
		Helper function for summary to extract summaries
		'''
		size = np.size(statdict['mean'])
		print_('\n%s:' % variable)
		print_(' ')
		buffer = []
		iindex = [key.split()[-1] for key in statdict.keys()].index('interval')
		interval = list(statdict.keys())[iindex]
		buffer += [
			'Mean             SD               MC Error        %s' %
			interval]
		buffer += ['-' * len(buffer[-1])]
		indices = range(size)
		if len(indices) == 1:
			indices = [None]
		for index in indices:
			m = str(round(statdict['mean'][index], roundto))
			sd = str(round(statdict['standard deviation'][index], roundto))
			mce = str(round(statdict['mc error'][index], roundto))
			hpd = str(statdict[interval][index].squeeze().round(roundto))
			valstr = m
			valstr += ' ' * (17 - len(m)) + sd
			valstr += ' ' * (17 - len(sd)) + mce
			valstr += ' ' * (len(buffer[-1]) - len(valstr) - len(hpd)) + hpd
			buffer += [valstr]
		buffer += [''] * 2
		buffer += ['Posterior quantiles:', '']
		buffer += [
			'2.5             25              50              75             97.5']
		buffer += [
			' |---------------|===============|===============|---------------|']
		for index in indices:
			quantile_str = ''
			for i, q in enumerate((2.5, 25, 50, 75, 97.5)):
				qstr = str(round(statdict['quantiles'][q][index], roundto))
				quantile_str += qstr + ' ' * (17 - i - len(qstr))
			buffer += [quantile_str.strip()]
		buffer += ['']
		print_('\t' + '\n\t'.join(buffer))

	def stats(self, variables=None, alpha=0.05, start=0,
			  batches=100, chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
		'''
		Statistical output for variables

		Parameters
		----------
		variables : list of str
			List of variable names to be summarized. By default it summarizes every variable
		alpha : float
			The alpha level for generating posterior intervals. Defaults to 0.05
		start : int
			The starting index from which to summarize chain. Defaults to zero
		batches : int 
			Batch size for calculating standard deviation for non-independent samples.
			Defaults to 100
		chain : int
			The index for which chain to summarize. Defaults to None (all chains)
		quantiles : tuple or list
			The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5)
		'''
		pattern = re.compile(".*adaptive_scale_factor|deviance")
		if variables is None:
			variables = [v for v in self._variables_to_tally if not pattern.match(v)]
		else:
			variables = [v for v in variables if (v in self._variables_to_tally) and (not pattern.match(v))]

		stat_dict = {}

		for variable in variables:
			stat_dict[variable] = self.trace(variable).stats(alpha=alpha, start=start,
															 batches=batches, chain=chain, quantiles=quantiles)
		return stat_dict

	def write_csv(self, filename, variables=None, alpha=0.05, start=0, batches=100,
				  chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
		'''
		Save summary statistics to a csv table on local machine

		Parameters
		----------
		filename: str
			Filename to save output
		variables : list of str
			List of variable names to be summarized. By default it summarizes every variable
		alpha : float
			The alpha level for generating posterior intervals. Defaults to 0.05
		start : int
			The starting index from which to summarize chain. Defaults to zero
		batches : int 
			Batch size for calculating standard deviation for non-independent samples.
			Defaults to 100
		chain : int
			The index for which chain to summarize. Defaults to None (all chains)
		'''
		if filename.find('.') == -1:
			filename += '.csv'
		outfile = open(filename, 'w')
		header = 'Parameter, Mean, SD, MC Error, Lower 95% HPD, Upper 95% HPD, '
		header += ', '.join(['q%s' % i for i in quantiles])
		outfile.write(header + '\n')
		stats = self.stats(variables=variables, alpha=alpha, start=start, 
						   batches=batches, chain=chain, quantiles=quantiles)
		if variables is None:
			variables = sorted(stats.keys())
		buffer = str()
		for param in variables:
			values = stats[param]
			try:
				shape = values['mean'].shape
				indices = list(itertools.product(*[range(i) for i in shape]))
				for i in indices:
					buffer += self._csv_str(param, values, quantiles, i)
			except AttributeError:
				buffer += self._csv_str(param, values, quantiles)
		outfile.write(buffer)
		outfile.close()

	def _csv_str(self, param, stats, quantiles, index=None):
		'''
		Helper function for write_csv
		'''
		buffer = param
		if not index:
			buffer += ', '
		else:
			buffer += '_' + '_'.join([str(i) for i in index]) + ', '

		for stat in ('mean', 'standard deviation', 'mc error'):
			buffer += str(stats[stat][index]) + ', '
		iindex = [key.split()[-1] for key in stats.keys()].index('interval')
		interval = list(stats.keys())[iindex]
		buffer += ', '.join(stats[interval][index].astype(str))
		qvalues = stats['quantiles']
		for q in quantiles:
			buffer += ', ' + str(qvalues[q][index])
		return buffer + '\n'

	def save_as_pickle_object(self, filename, batch_size=10, chain=None):
		'''
		Save the data to HDFS as pickle objects
		Requires Spark v1.0

		Parameters
		----------
		filename : str
			Name of the file to save the object
		batch_size : int
			Batch size for serializing the data
		chain : int
			The index for which chain to summarize. Defaults to None (all chains)
		'''
		if chain is None:
			self.db.rdd.saveAsPickleFile(filename, batch_size)
		else:
			if chain < 0:
				chain = xrange(self.db.chains)[chain]
			self.db.rdd.filter(lambda x: x[0] == chain).saveAsPickleFile(filename, batch_size)

	def save_as_txt_file(self, path, chain=None):
		'''
		Save the data to HDFS as txt files
		
		Parameters
		----------
		path : str
			Name of the file to save the data
		chain : int
			The index for which chain to summarize. Defaults to None (all chains)
		'''
		temp_rdd = self.db.rdd
		if chain is not None:
			if chain < 0:
				chain = xrange(self.db.chains)[chain]
			temp_rdd = self.db.rdd.filter(lambda x: x[0]==chain).cache()
		for v in self._variables_to_tally:
			def save_mapper(x):
				data = '# Variable: %s\n' % v
				data += '# Sample shape: %s\n' % str(x.shape)
				data += '# Date: %s\n' % datetime.datetime.now()
				X = x.reshape((-1, x[0].size))
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
			temp_rdd.map(lambda x: x[1][v]).map(save_mapper).saveAsTextFile(os.path.join(path, v))
		temp_rdd.map(lambda x: x[1]['_state_']).saveAsTextFile(os.path.join(path, 'state'))

	def remember(self, chain=-1, trace_index=None):
		pass
