'''
Test MCMC on Disaster Model on Spark
'''
from __future__ import with_statement
from pyspark import SparkContext
from pymc.MCMCSpark import MCMCSpark
from pymc.examples import disaster_model
from numpy.testing import *
import nose
import warnings
import numpy as np


class test_MCMCSpark_withHDFS(TestCase):

    @classmethod
    def setUpClass(self):
        self.M = MCMCSpark(input=disaster_model, nJobs=10)

    def test_sample(self):
        self.M.sample(50, 25, 5, progress_bar=0)
        assert hasattr(self.M.db, '__Trace__')
        assert hasattr(self.M.db, '__name__')
        for chain in xrange(10):
            assert_array_equal(
                self.M.trace('early_mean', chain=chain)[:].shape, (5,))
            assert_equal(self.M.trace('early_mean', chain=chain).length(), 5)
            assert_equal(
                self.M.trace('early_mean',
                             chain=chain)[:].__class__,
                np.ndarray)
            assert_equal(self.M.trace('early_mean', chain)._chain, chain)

        assert_array_equal(self.M.trace('early_mean')[:].shape, (5,))
        assert_equal(self.M.trace('early_mean').length(), 5)
        assert_array_equal(
            self.M.trace('early_mean', chain=None)[:].shape, (50,))
        assert_equal(self.M.trace('early_mean').length(chain=None), 50)
        assert_equal(self.M.trace('early_mean').gettrace(
            slicing=slice(1, 2)), db.trace('early_mean')[1])

if __name__ == '__main__':
    original_filters = warnings.filters[:]
    warnings.simplefilter("ignore")
    try:
        import nose
        nose.runmodule()
    finally:
        warnings.filters = original_filters
