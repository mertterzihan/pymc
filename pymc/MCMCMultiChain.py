'''
Class for performing distributed Multi Chain MCMC, extends MCMC
'''

__all__ = ['MCMCMultiChain']

from .MCMC import MCMC
from . import database

class MCMCMultiChain(MCMC):
	def __init__(self, input=None, db='ram', name='MCMC', calc_deviance=True, chainNum=-1, final_job=True, **kwds):
		self.chainNum = chainNum
		self.final_job = final_job
		MCMC.__init__(self, input, db, name, calc_deviance, **kwds)

	def _finalize(self):
		"""Reset the status and tell the database to finalize the traces."""
		if self.status in ['running', 'halt']:
			if self.verbose > 0:
				print_('\nSampling finished normally.')
			self.status = 'ready'
		if self.final_job:
			self.save_state()
		self.db._finalize()

	def _assign_database_backend(self, db):
		'''
		Assign Trace instance to stochastics and deterministics and Database instance
		to self.

		:Parameters:
		  - `db` : string, Database instance
		    The name of the database module (see below), or a Database instance.

		Available databases:
		  - 'hdfs': traces are stored as text files on HDFS
		'''
		no_trace = getattr(database, 'no_trace')
		self._variables_to_tally = set()
		for object in self.stochastics | self.deterministics:

			if object.keep_trace:
				self._variables_to_tally.add(object)
				try:
					if object.mask is None:
						# Standard stochastic
						self._funs_to_tally[object.__name__] = object.get_value
					else:
						# Has missing values, so only fetch stochastic elements
						# using mask
						self._funs_to_tally[
						    object.__name__] = object.get_stoch_value
				except AttributeError:
					# Not a stochastic object, so no mask
					self._funs_to_tally[object.__name__] = object.get_value
			else:
				object.trace = no_trace.Trace(object.__name__)

		check_valid_object_name(self._variables_to_tally)

		# If not already done, load the trace backend from the database
		# module, and assign a database instance to Model.
		if isinstance(db, str):
			if db in dir(database):
				module = getattr(database, db)

				# Assign a default name for the database output file.
				if self._db_args.get('dbname') is None:
					self._db_args['dbname'] = self.__name__ + '.' + db
				if self.chainNum is not -1:
					self.db = module.Database(chainFile=self.chainNum, **self._db_args)
				else:
					self.db = module.Database(**self._db_args)
			elif db in database.__modules__:
				raise ImportError(
				'Database backend `%s` is not properly installed. Please see the documentation for instructions.' % db)
			else:
				raise AttributeError(
				    'Database backend `%s` is not defined in pymc.database.' % db)
		elif isinstance(db, database.base.Database):
			self.db = db
			self.restore_sampler_state()
		else:   # What is this for? DH.
			self.db = db.Database(**self._db_args)

def check_valid_object_name(sequence):
	"""Check that the names of the objects are all different."""
	names = []
	for o in sequence:
		if o.__name__ in names:
			raise ValueError(
				'A tallyable PyMC object called %s already exists. This will cause problems for some database backends.' %
				o.__name__)
		else:
			names.append(o.__name__)

