'''
HDFS trace backends

After sampling with NDArray backend, save results in HDFS as text files.

Sampling can be performed via passing an HDFS backend instance to 'sample':
	>>> import pymc as pm
	>>> hdfs = pm.backends.HDFS(name='user/test', host='localhost', port='50070', user_name='test')
	>>> trace = pm.sample(..., trace=hdfs)

Loading a saved trace can be accomplished as follows:
	>>> import pymc as pm
	>>> trace = pm.backends.hdfs.load(name='user/test', host='localhost', port='50070', user_name='test')

Database format
---------------
For each chain, a directory named `chain-N` is created in the specified 
directory in HDFS. In this directory, one file per variable is created 
containing the values of the object. To deal with multidimensional 
variables, the array is reshaped to one dimension before saving with 
'save_nparray_to_hdfs', an instance of numpy's savetxt function to 
enable saving txt files on HDFS. The shape information is saved in a 
json file in the same directory and is used to load the database back 
again using `numpy.loadtxt`.
'''

import os
import json
import re
import numpy as np
import StringIO

from pymc.backends import base
from pymc.backends.ndarray import NDArray
from pywebhdfs.webhdfs import PyWebHdfsClient
from numpy.compat import asstr, asbytes
from pywebhdfs.errors import FileNotFound

class HDFS(NDArray):
	'''
	HDFS storage

	Parameters
	----------
	name : str
		Name of directory to store text files (Path to the directory) without
		a leading '/'
	model : Model
		If None, the model is taken from the 'with' context
	vars : list of variables
		Sampling values will be stored for these variables. If None.
		'model.unobserved_RVs' is used
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
	def __init__(self, name, model=None, vars=None, host='localhost', port='50070', user_name=None):
		self.hdfs = PyWebHdfsClient(host=host, port=port, user_name=user_name)
		try:
			self.hdfs.list_dir(name)
		except FileNotFound:
			self.hdfs.make_dir(name)
		super(HDFS, self).__init__(name, model, vars)

	def close(self):
		super(HDFS, self).close()
		_dump_trace(self.name, self)


def dump(name, trace, chains=None):
	'''
	Store NDArray trace in HDFS as text database

	Parameters
	----------
	name : str
		Path to root directory for text database without a leading '/'
	trace : MultiTrace of NDArray traces
		Result of MCMC run with default NDArray backend
	chains : list 
		Chains to dump. If None, all chains are dumped
	'''
	try:
		trace.hdfs.list_dir(name)
	except FileNotFound:
		trace.hdfs.make_dir(name)
	if chains is None:
		chains = trace.chains
		for chain in chains:
			_dump_trace(name, trace._traces[chain])


def _dump_trace(name, trace):
	'''
	Dump a single-chain trace
	'''
	chain_name = 'chain-{}'.format(trace.chain)
	chain_dir = os.path.join(name, chain_name)
	try:
		trace.hdfs.list_dir(chain_dir)
	except FileNotFound:
		trace.hdfs.make_dir(chain_dir)

	shapes = {}
	for varname in trace.varnames:
		data = trace.get_values(varname)
		var_file = os.path.join(chain_dir, varname + '.txt')
		save_nparray_to_hdfs(var_file, data.reshape(-1, data.size), trace.hdfs)
		shapes[varname] = data.shape
	shape_file = os.path.join(chain_dir, 'shapes.json')
	shape_data = json.dumps(shapes)
	trace.hdfs.create_file(shape_file, shape_data, overwrite=True)


def save_nparray_to_hdfs(fname, X, hdfs):
	'''
	An instance of numpy's savetext function to enable saving numpy 
	arrays in HDFS as text files
	'''
	fmt = '%.18e'
	delimiter = ' '
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
	first = True
	for row in X:
		if first:
			hdfs.create_file(fname, asbytes(format % tuple(row) + newline), overwrite=True)
			first = False
		else:	
			hdfs.appendfile(fname, asbytes(format % tuple(row) + newline))


def load(name, chains=None, model=None, host='localhost', port='50070', user_name=None):
	'''
	Load text database

	Parameters
	----------
	name : str
		Path to root directory in HDFS for text database without a leading '/'
	chains : list
		Chains to load. If None, all chains are loaded
	model : Model
		If None, the model is taken from the 'with' context
	host : str
		The IP address or hostname of the HDFS namenode. By default,
		it is 'localhost'
	port : str
		The port number for WebHDFS on the namenode. By default, it
		is '50070'
	user_name : str
		WebHDFS user_name used for authentication. By default, it is
		None

	Returns
	-------
	ndarray.Trace instance
	'''
	hdfs = PyWebHdfsClient(host=host, port=port, user_name=user_name)
	chain_dirs = _get_chain_dirs(name, hdfs)
	if chains is None:
		chains = list(chain_dirs.keys())
	traces = []
	for chain in chains:
		chain_dir = chain_dirs[chain]
		dir_path = os.path.join(name, chain_dir)
		shape_file = os.path.join(dir_path, 'shapes.json')
		shapes = json.load(StringIO.StringIO(hdfs.read_file(shape_file)))
		samples = {}
		for varname, shape in shapes.items():
			var_file = os.path.join(dir_path, varname + '.txt')
			samples[varname] = np.loadtxt(StringIO.StringIO(str(hdfs.read_file(var_file)))).reshape(shape)
		trace = NDArray(model=model)
		trace.samples = samples
		trace.chain = chain
		traces.append(trace)
	return base.MultiTrace(traces)


def _get_chain_dirs(name, hdfs):
	'''
	Return mapping of chain number to directory
	'''
	data = hdfs.list_dir(name)
	regex = re.compile("chain-*")
	chain_dict = {}
	for d in data['FileStatuses']['FileStatus']:
		chain_name = str(d['pathSuffix'])
		if regex.match(chain_name):
			chain_dict[int(chain_name.split('-')[1])] = chain_name
	return chain_dict
