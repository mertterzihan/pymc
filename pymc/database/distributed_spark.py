'''
Distributed Spark backend

Store the traces of DistributedMCMC on Spark RDDs as dictionaries
'''

import numpy as np
from pymc.utils import make_indices, calc_min_interval
import os

__all__ = ['Trace', 'Database']

class Trace():
	'''
	Distributed Spark Trace
	'''
	def __init__(self, name, db=None, chain=-1):
		self.name = name
		self.db = db
		self._chain = chain

	def truncate(self, index, chain):
		'''
		Truncate the trace array to some index

		Parameters
		----------
		index : int
			The index within the chain after which all values will be removed
		chain : int
			The chain index (>=0)
		'''
		tname = self.name
		def truncate_helper(x):
			if tname in x[1][chain]:
				x[1][chain][tname] = x[1][tname][:index]
			return x
		new_rdd = self.db.rdd.map(truncate_helper).cache()
		self.db.rdd = new_rdd

	def gettrace(self, burn=0, thin=1, chain=-1, slicing=None):
		'''
		Return the trace

		Parameters
		----------
		burn : int
			The number of transient steps to skip
		thin : int
			Keep one in thin
		chain : int
			The index of the chain to fetch. If None, return all chains
		slicing : slice
			A slice, overriding burn and thin assignments
		'''
		tname = self.name
		if slicing is None:
			slicing = slice(burn, None, thin)
		if chain is not None:
			if chain < 0:
				chain = xrange(self.db.chains)[chain]
			result = self.db.rdd.filter(lambda x: tname in x[1][chain]).map(lambda x: x[1][chain][tname][slicing]).collect()
			if type(result) is list and len(result) == 1:
				return result[0]
			else:
				return result
			# return self.db.rdd.filter(lambda x: x[0]==chain and tname in x[1]).map(lambda x: x[1][tname][slicing]).collect()
		else:
			def map_helper(x):
				from numpy import concatenate
				result = x[1][0][tname][slicing]
				for collection in x[1][1:]:
					result = concatenate(result, collection[tname][slicing])
				return result
			result = self.db.rdd.filter(lambda x: tname in x[1][0]).map(map_helper).collect()
			if type(result) is list and len(result) == 1:
				return result[0]
			else:
				return result

	def __getitem__(self, index):
		chain = self._chain
		tname = self.name
		if chain is None:
			def map_helper(x):
				from numpy import concatenate
				result = x[1][0][tname][index]
				for collection in x[1][1:]:
					result = concatenate(result, collection[tname][index])
				return result
			result = self.db.rdd.filter(lambda x: tname in x[1][0]).map(map_helper).collect()
			if type(result) is list and len(result) == 1:
				return result[0]
			else:
				return result
		else:
			if chain < 0:
				chain = range(self.db.chains)[chain]
			result = self.db.rdd.filter(lambda x: tname in x[1][chain]).map(lambda x: x[1][chain][tname][index]).collect()
			if type(result) is list and len(result) == 1:
				return result[0]
			else:
				return result

	__call__ = gettrace

	def length(self, chain=-1):
		'''
		Return the length of the trace

		Parameters
		----------
		chain : int
			The chain index. If None, returns the combined length of all chains
		'''
		from operator import add
		tname = self.name
		if chain is not None:
			if chain < 0:
				chain = range(self.db.chains)[chain]
			return self.db.rdd.filter(lambda x: tname in x[1][chain]).map(lambda x: x[1][chain][tname].shape[0]).reduce(add)
		else:
			def map_helper(x):
				total_length = 0
				for collection in x[1]:
					total_length += collection[tname].shape[0]
				return total_length
			return self.db.rdd.filter(lambda x: tname in x[1][0]).map(map_helper).reduce(add)

	def stats(self, alpha=0.05, start=0, batches=100,
			  chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
		'''
		Generate posterior statistics for node

		Parameters
		----------
		name : str 
			The name of the tallyable object
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
		stat = dict()
		tname = self.name
		#stat['n'] = self.length(chain=chain)
		filtered_rdd = None
		if chain is None:
			def concat(x):
				return np.concatenate([chain[tname] for chain in x[1]])[start:]
			filtered_rdd = self.db.rdd.filter(lambda x: tname in x[1][0]).map(concat).cache()
		else:
			if chain < 0:
				chain = xrange(self.db.chains)[chain]
			filtered_rdd = self.db.rdd.filter(lambda x: tname in x[1][chain]).map(lambda x: np.squeeze(np.array(x[1][chain][tname]))[start:]).cache()
		result = self.calc_mean_std(filtered_rdd)
		stat['n'] = result[0]
		stat['mean'] = result[1]
		stat['standard deviation'] = result[2]
		stat['mc error'] = self.calc_batchsd(filtered_rdd, batches)
		stat['quantiles'] = self.calc_quantile(filtered_rdd, quantiles)
		stat['%s%s HPD interval' % (int(100 * (1 - alpha)), '%')] = self.calc_hpd(filtered_rdd, alpha)
		return stat

	def calc_mean_std(self, rdd):
		'''
		Helper function to calculate length, mean and standard deviation
		'''
		def mapper_helper(x):
			return x.shape[0], x.mean(0), x.std(0)
		def reduce_helper(x, y):
			weighted_mean = np.divide(np.multiply(x[1], x[0]) + np.multiply(y[1], y[0]), x[0]+y[0], dtype=float)
			weighted_std = np.add(np.multiply(np.square(x[2]), x[0]-1), np.multiply(np.square(y[2]), y[0]-1))
			weighted_std = np.add(weighted_std, np.multiply(np.square(x[1]), x[0]))
			weighted_std = np.add(weighted_std, np.multiply(np.square(y[1]), y[0]))
			weighted_std = np.subtract(weighted_std, np.multiply(np.square(weighted_mean), x[0]+y[0]))
			weighted_std = np.divide(weighted_std, x[0]+y[0]-1, dtype=float)
			return (x[0]+y[0], weighted_mean, np.sqrt(weighted_std))
		return rdd.map(mapper_helper).reduce(reduce_helper)

	def calc_batchsd(self, rdd, batches=5):
		'''
		Helper function to calculate mc error
		'''
		def map_helper(trace):
			if not isinstance(trace, np.ndarray) or batches > len(trace):
				return 'Could not generate mc error'
			if len(np.shape(trace)) > 1:
				dims = np.shape(trace)
				ttrace = np.transpose([t.ravel() for t in trace])
				sd_list = [batchsd_helper(t) for t in ttrace]
				return (np.reshape([t[0] for t in sd_list], dims[1:]), np.reshape([t[1] for t in sd_list], dims[1:]), sd_list[0][2])
			else:
				if batches == 1:
					return np.std(trace) / np.sqrt(len(trace))
				try:
					batched_traces = np.resize(trace, (batches, len(trace) / batches))
				except ValueError:
					# If batches do not divide evenly, trim excess samples
					resid = len(trace) % batches
					batched_traces = np.resize(
						trace[:-resid],
						(batches,
						 len(trace) / batches))
			means = np.mean(batched_traces, 1)
			return (np.std(means), np.mean(means), len(trace)/batches)
		def reduce_helper(x, y):
			if isinstance(x, str) or isinstance(y,str):
				return 'Could not generate mc error'
			weighted_mean = np.divide(np.add(np.multiply(x[1], x[2]), np.multiply(y[1], y[2])), np.add(x[2],y[2]), dtype=float)
			weighted_std = np.add(np.multiply(np.square(x[0]), x[2]-1), np.multiply(np.square(y[0]), y[2]-1))
			weighted_std = np.add(weighted_std, np.multiply(np.square(x[1]), x[2]))
			weighted_std = np.add(weighted_std, np.multiply(np.square(y[1]), y[2]))
			weighted_std = np.subtract(weighted_std, np.multiply(np.square(weighted_mean), x[2]+y[2]))
			weighted_std = np.divide(weighted_std, x[2]+y[2]-1, dtype=float)
			return (weighted_std, weighted_mean, x[2]+y[2])
		result = rdd.map(map_helper).reduce(reduce_helper)
		if isinstance(result, str):
			return result
		else:
			return result[0] / np.sqrt(batches)

	def calc_quantile(self, rdd, quantiles):
		'''
		Helper function to calculate quantiles
		'''
		def map_helper(x):
			if x.ndim > 1:
				sx = np.sort(x.T).T
			else:
				sx = np.sort(x)
			try: 
				quants = [sx[int(len(sx) * q / 100.0)] for q in quantiles]
				return (1, dict(zip(quantiles, quants)))
			except IndexError:
				return 'Too few elements for quantile calculation'
		def reduce_helper(x, y):
			average_quantile = dict()
			for k in x[1].keys():
				average_quantile[k] = np.divide(np.add(np.multiply(x[1][k], x[0]), np.multiply(y[1][k], y[0])), x[0]+y[0], dtype=float)
			return (x[0]+y[0], average_quantile)
		return rdd.map(map_helper).reduce(reduce_helper)[1]

	def calc_hpd(self, rdd, alpha=0.05):
		'''
		Helper function to calculate HPD interval
		'''
		def hpd_map_helper(x):
			if x.ndim > 1:
				tx = np.transpose(x, list(range(x.ndim)[1:]) + [0])
				dims = np.shape(tx)
				intervals = np.resize(0.0, dims[:-1] + (2,))
				for index in make_indices(dims[:-1]):
					try:
						index = tuple(index)
					except TypeError:
						pass
					sx = np.sort(tx[index])
					intervals[index] = calc_min_interval(sx, alpha)
				return (1, np.array(intervals))
			else:
				sx = np.sort(x)
				return (1, np.array(calc_min_interval(sx, alpha)))
		def hpd_reduce_helper(x, y):
			return (x[0]+y[0], np.divide(np.add(np.multiply(x[1], x[0]), np.multiply(y[1], y[0])), x[0]+y[0]))
		return rdd.map(hpd_map_helper).reduce(hpd_reduce_helper)[1]


class Database():
	'''
	Distributed Spark Database
	'''

	def __init__(self, rdd, funs_to_tally):
		self.__Trace__ = Trace
		self.__name__ = 'distributed_spark'
		self.trace_names = funs_to_tally
		self.rdd = rdd
		self._traces = {}
		self.chains = self.rdd.map(lambda x: len(x[1])).first()
		for tname in self.trace_names:
			if tname not in self._traces:
				self._traces[tname] = self.__Trace__(name=tname, db=self, chain=self.chains)

	def trace(self, name, chain=-1):
		'''
		Return the trace of a tallyable object stored in the database

		Parameters
		----------
		name : str
			The name of the tallyable object
		chain : int
			The trace index. Setting 'chain=i' will return the trace created by the ith spark job
		'''
		trace = self._traces[name]
		trace._chain = chain
		return trace

	def getstate(self):
		'''
		Return a list of dictionaries, each containing the state of the Model and its StepMethods of its corresponding chain
		'''
		chain = self.chains-1
		return self.rdd.map(lambda x: (x[0], x[1][chain]['_state_'])).collect()

	def truncate(self, index, chain=-1):
		'''
		Tell the traces to truncate themselves at the given index

		Parameters
		----------
		index : int
			The index within the chain after which all values will be removed
		chain : int
			Chain index
		'''
		chain = range(self.chains)[chain]
		def truncate_helper(x):
			trace_names = x[1][chain].keys()
			for tname in trace_names:
				x[1][chain][tname] = x[1][chain][tname][:index]
			return x
		new_rdd = self.rdd.map(truncate_helper).cache()
		self.rdd = new_rdd


def load_pickle(spark_context, dbname):
	pass


def load_txt(spark_context, dbname):
	pass
		