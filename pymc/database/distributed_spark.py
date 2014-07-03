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
			if x[0] == chain and tname in x[1]:
				x[1][tname] = x[1][tname][:index]
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
			return self.db.rdd.filter(lambda x: x[0]==chain and tname in x[1]).map(lambda x: x[1][tname][slicing]).collect()
		else:
			def reduce_helper(x, y):
				from numpy import concatenate
				return (x[0], concatenate([x[1],y[1]]))
			return self.db.rdd.filter(lambda x: tname in x[1]).map(lambda x: (x[0], x[1][tname][slicing])).sortByKey().reduce(reduce_helper)[1]

	def __getitem__(self, index):
		chain = self._chain
		tname = self.name
		if chain is None:
			def reduce_helper(x, y):
				from numpy import concatenate
				return (x[0], concatenate(x[1], y[1]))
			return self.db.rdd.filter(lambda x: tname in x[1]).map(lambda x: (x[0], x[1][tname][index])).sortByKey().reduce(reduce_helper)[1]
		else:
			if chain < 0:
				chain = range(self.db.chain)[chain]
			return self.db.rdd.filter(lambda x: x[0]==chain and tname in x[1]).map(lambda x: x[1][tname][index]).collect()

	__call__ = gettrace

	def length(self, chain=-1):
		tname = self.name
		if chain is not None:
			if chain < 0:
				chain = range(self.db.chains)[chain]
			return self.db.rdd.filter(lambda x: x[0]==chain and tname in x[1]).map(lambda x: x[1][tname].shape[0]).collect()
		else:
			from operator import add
			return self.db.rdd.filter(lambda x: tname in x[1]).map(lambda x: x[1][tname].shape[0]).reduce(add)

	def stats(self, alpha=0.05, start=0, batches=100,
			  chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
		pass


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
		self.chains = self.rdd.count()
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
		return self.rdd.map(lambda x: (x[0], x[1]['_state_'])).collect()

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
			trace_names = x[1].keys()
			if x[0] == chain:
				for tname in trace_names:
					x[1][tname] = x[1][tname][:index]
			return x
		new_rdd = self.rdd.map(truncate_helper).cache()
		self.rdd = new_rdd


def load_pickle(spark_context, dbname):
	pass


def load_txt(spark_context, dbname):
	pass
		