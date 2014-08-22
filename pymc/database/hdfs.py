from __future__ import with_statement
'''
HDFS database module

Store the traces in txt files on HDFS.

For each chain, a directory named `Chain_#` is created. In this directory,
one file per tallyable object is created containing the values of the object.

An example of HDFS's usage can be found under examples (disaster_model_hdfs.py)

Implementation Notes
--------------------
The NumPy arrays are saved by an adaptation of NumPy's 'savetxt' and loaded 
using NumPy's 'loadtxt'.

Changeset
---------
'''


from . import base, ram
import os
import datetime
import re
import StringIO
import numpy as np
from numpy import array
from numpy.compat import asbytes, asstr

from pymc import six
from pywebhdfs.webhdfs import PyWebHdfsClient
from pywebhdfs.errors import FileNotFound

__all__ = ['Trace', 'Database', 'load']

CHAIN_NAME = 'Chain_%d'

class Trace(ram.Trace):
	'''
	HDFS Trace Class

	Store the trace in txt files on HDFS in a specific directory to chain

	dbname/
      Chain_0/
        <object name>.txt
        <object name>.txt
        ...
      Chain_1/
        <object name>.txt
        <object name>.txt
        ...
      ...
	'''

	def _finalize(self, chain):
		'''
		Write trace to a txt file on HDFS

		Parameters
		----------
		chain : int
			The chain index
		'''
		path = os.path.join(
            self.db._directory,
            self.db.get_chains()[chain],
            self.name + '.txt')
		arr = self.gettrace(chain=chain)
		self.db.hdfs.create_file(path, '# Variable: %s\n' % self.name, overwrite=True)
		self.db.hdfs.append_file(path, '# Sample shape: %s\n' % str(arr.shape))
		self.db.hdfs.append_file(path, '# Date: %s\n' % datetime.datetime.now())
		save_nparray_to_hdfs(path, arr.reshape((-1, arr[0].size)), self.db.hdfs)


class Database(base.Database):
	'''
	HDFS Database class
	'''

	def __init__(self, dbname=None, host='localhost', port='50070', user_name=None):
		'''
		Create HDFS Database

		Parameters
		----------
		dbname : string
			Name of directory where the traces are stored without a 
			leading '/'
		host : str
			The IP address or hostname of the HDFS namenode. By default,
			it is 'localhost'
		port : str
			The port number for WebHDFS on the namenode. By default, it
			is '50070'
		user_name : str
			WebHDFS user_name used for authentication. By default, it is
			None
		'''
		self.__name__ = 'hdfs'
		self._directory = dbname
		self.__Trace__ = Trace
		self.hdfs = PyWebHdfsClient(host=host, port=port, user_name=user_name)

		self.trace_names = []
		self._traces = {}
		self.chains = 0

		try:
			self.hdfs.list_dir(self._directory)
		except FileNotFound:
			self.hdfs.make_dir(self._directory)

	def get_chains(self):
		'''
		Return an ordered list of the 'Chain_#' directories in the db directory
		'''
		chains = []
		data = self.hdfs.list_dir(self._directory)
		regex = re.compile(CHAIN_NAME[:-2] + '*')
		for d in data['FileStatuses']['FileStatus']:
			file_name = str(d['pathSuffix'])
			if regex.match(file_name):
				chains.append(file_name)
		chains.sort()
		return chains

	def _initialize(self, funs_to_tally, length):
		'''
		Create folder to store simulation results
		'''
		dir = os.path.join(self._directory, CHAIN_NAME % self.chains)
		self.hdfs.make_dir(dir)

		base.Database._initialize(self, funs_to_tally, length)

	def savestate(self, state):
		'''
		Save the sampler's state in a state.txt file
		'''
		file_name = os.path.join(self._directory, 'state.txt')
		print_state(state, file_name=file_name, hdfs=self.hdfs)

def load(dirname, host='localhost', port='50070', user_name=None):
	'''
	Create a Database instance from the data stored in a directory on HDFS

	Parameters
	----------
	dirname : string
		Name of directory where the traces are stored without a 
		leading '/'
	host : str
		The IP address or hostname of the HDFS namenode. By default,
		it is 'localhost'
	port : str
		The port number for WebHDFS on the namenode. By default, it
		is '50070'
	user_name : str
		WebHDFS user_name used for authentication. By default, it is
		None
	'''
	hdfs = PyWebHdfsClient(host=host, port=port, user_name=user_name)
	try:
		hdfs.list_dir(dirname)
	except FileNotFound:
		raise AttributeError('No txt database named %s' % dirname)
	db = Database(dirname, host=host, port=port, user_name=user_name)
	chain_folders = [os.path.join(dirname, c) for c in db.get_chains()]
	db.chains = len(chain_folders)

	data = {}
	for chain, folder in enumerate(chain_folders):
		status = hdfs.list_dir(folder)
		files = []
		for s in status['FileStatuses']['FileStatus']:
			files.append(str(s['pathSuffix']))
		funnames = funname(files)
		db.trace_names.append(funnames)
		for file in files:
			name = funname(file)
			if name not in data:
				data[name] = {}
			data_stream = StringIO.StringIO(hdfs.read_file(os.path.join(folder, file)))
			data_stream.readline()
			shape = eval(data_stream.readline()[16:])
			data[name][chain] = np.loadtxt(data_stream,delimiter=',').reshape(shape)
	# Create the Traces.
	for name, values in six.iteritems(data):
		db._traces[name] = Trace(name=name, value=values, db=db)
		setattr(db, name, db._traces[name])
	# Load the state.
	statefile = os.path.join(dirname, 'state.txt')
	try:
		f = StringIO.StringIO(hdfs.read_file(statefile))
		db._state_ = eval(f.read())
	except FileNotFound:
		db._state_ = {}

	return db


def funname(file):
    """Return variable names from file names."""
    if isinstance(file, str):
        files = [file]
    else:
        files = file
    bases = [os.path.basename(f) for f in files]
    names = [os.path.splitext(b)[0] for b in bases]
    if isinstance(file, str):
        return names[0]
    else:
        return names

def save_nparray_to_hdfs(fname, X, hdfs):
	'''
	An instance of numpy's savetext function to enable saving numpy 
	arrays in HDFS as text files
	'''
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
	output_strings = [asbytes(format % tuple(row) + newline) for row in X]
	hdfs.append_file(fname, ''.join(output_strings))

def print_state(*args, **kwargs):
	'''
	Editing six.print_ to make it work with HDFS
	'''
	file_name = kwargs.pop("file_name", None)
	hdfs = kwargs.pop("hdfs", None)
	if not all([file_name, hdfs]):
		return
	def write(data, hdfs, firstline):
		if not isinstance(data, basestring):
			data = str(data)
		if firstline:
			hdfs.create_file(file_name, data, overwrite=True)
			firstline = False
		else:
			hdfs.append_file(file_name, data)
		return firstline
	end = "\n"
	sep = " "
	firstline = True
	for i, arg in enumerate(args):
		if i:
			firstline = write(sep, hdfs, firstline)
		firstline = write(arg, hdfs, firstline)
	firstline = write(end, hdfs, firstline)