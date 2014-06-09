'''
Test MCMC on Disaster Model on Spark
'''
from __future__ import with_statement
from pymc.MCMCSpark import MCMCSpark
from pymc import database
from numpy.testing import *
import nose
import warnings
import numpy as np

spark_host = 'local[4]'
spark_home = '/Users/mert.terzihan/Documents/Spark/spark'
model_file = '/Users/mert.terzihan/Documents/pymc/pymc/examples/disaster_model.py'
dbname = 'user/mert.terzihan/unittest/test4/spark'
hdfs_host = 'bottou01-10g.pa.cloudera.com'
port = '50070'
user_name = 'mert.terzihan'

class test_MCMCSpark_withHDFS(TestCase):
	@classmethod
	def setUpClass(self):
		self.M = MCMCSpark(db='hdfs', dbname=dbname, model_file=model_file, spark_home=spark_home, spark_host=spark_host, nJobs=10, hdfs_host=hdfs_host, port=port, user_name=user_name)

	def test_sample(self):
		db = self.M.sample(50, 25, 5, progress_bar=0)
		assert hasattr(db, '__Trace__')
		assert hasattr(db, '__name__')
		for chain in xrange(10):
			assert_array_equal(db.trace('early_mean', chain=chain)[:].shape, (5,))
			assert_equal(db.trace('early_mean', chain=chain).length(), 5)
			assert_equal(db.trace('early_mean', chain=chain)[:].__class__, np.ndarray)

		assert_array_equal(db.trace('early_mean')[:].shape, (5,))
		assert_equal(db.trace('early_mean').length(), 5)
		assert_array_equal(db.trace('early_mean', chain=None)[:].shape, (50,))
		assert_equal(db.trace('early_mean').length(chain=None), 50)
		assert_equal(db.trace('early_mean').gettrace(slicing=slice(1, 2)), db.early_mean.trace[1])

if __name__ == '__main__':
	original_filters = warnings.filters[:]
	warnings.simplefilter("ignore")
	try:
		import nose
		nose.runmodule()
	finally:
		warnings.filters = original_filters