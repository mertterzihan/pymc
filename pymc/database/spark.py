'''
Spark backend

Store the traces on Spark RDDs as dictionaries
'''

import numpy as np
from pymc.utils import make_indices, calc_min_interval
import os
import re

__all__ = ['Trace', 'Database', 'load_pickle', 'load_txt']


class Trace():

    '''
    Spark Trace
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
            if x[0] == chain:
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
            return self.db.rdd.filter(lambda x: x[0] == chain).map(lambda x: x[1][tname][slicing]).first()
        else:
            def reduce_helper(x, y):
                from numpy import concatenate
                return (x[0], concatenate([x[1], y[1]]))
            return self.db.rdd.map(lambda x: (x[0], x[1][tname][slicing])).sortByKey().reduce(reduce_helper)[1]

    def __getitem__(self, index):
        chain = self._chain
        tname = self.name
        if chain is None:
            def reduce_helper(x, y):
                from numpy import concatenate
                return (x[0], concatenate([x[1], y[1]]))
            return self.db.rdd.map(lambda x: (x[0], x[1][tname][index])).sortByKey().reduce(reduce_helper)[1]
        else:
            if chain < 0:
                chain = range(self.db.chains)[chain]
            return self.db.rdd.filter(lambda x: x[0] == chain).map(lambda x: x[1][tname][index]).first()

    __call__ = gettrace

    def length(self, chain=-1):
        '''
        Return the length of the trace

        Parameters
        ----------
        chain : int
                The chain index. If None, returns the combined length of all chains
        '''
        tname = self.name
        if chain is not None:
            if chain < 0:
                chain = range(self.db.chains)[chain]
            return self.db.rdd.filter(lambda x: x[0] == chain).map(lambda x: x[1][tname].shape[0]).first()
        else:
            from operator import add
            return self.db.rdd.map(lambda x: x[1][tname].shape[0]).reduce(add)

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
        stat['n'] = self.length(chain=None)
        if chain is None:
            result = self.calc_mean_std()
            stat['mean'] = result[0]
            stat['standard deviation'] = result[1]
            stat['mc error'] = self.multiplechain_batchsd(batches=batches)
            stat['quantiles'] = self.calc_quantile_multchain(quantiles)
            stat['%s%s HPD interval' %
                 (int(100 * (1 - alpha)), '%')] = self.calc_hpd_multchain(alpha)
        else:
            if chain < 0:
                chain = xrange(self.db.chains)[chain]
            filtered_rdd = self.db.rdd.filter(
                lambda x: x[0] == chain).map(lambda x: x[1][tname]).cache()
            stat['mean'] = filtered_rdd.map(lambda x: x.mean(0)).first()
            stat['standard deviation'] = filtered_rdd.map(
                lambda x: x.std(0)).first()
            stat['mc error'] = self.batchsd(
                chain=chain, batches=batches, rdd=filtered_rdd)

            def quantile_map_helper(trace):
                x = trace.copy()
                if x.ndim > 1:
                    sx = np.sort(x.T).T
                else:
                    sx = np.sort(x)
                try:
                    quants = [sx[int(len(sx) * q / 100.0)] for q in quantiles]
                    return dict(zip(quantiles, quants))
                except IndexError:
                    return 'Too few elements for quantile calculation'
            stat['quantiles'] = filtered_rdd.map(quantile_map_helper).first()

            def hpd_map_helper(trace):
                x = trace.copy()
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
                    return np.array(intervals)
                else:
                    sx = np.sort(x)
                    return np.array(calc_min_interval(sx, alpha))
            stat['%s%s HPD interval' % (
                int(100 * (1 - alpha)), '%')] = filtered_rdd.map(hpd_map_helper).first()
        return stat

    def calc_hpd_multchain(self, alpha=0.05):
        '''
        Helper function to calculate HDP for multi-chain statistics
        '''
        tname = self.name

        def hpd_map_helper(trace):
            x = trace.copy()
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
            return (x[0] + y[0], np.divide(np.add(np.multiply(x[1], x[0]), np.multiply(y[1], y[0])), x[0] + y[0]))
        return self.db.rdd.map(lambda x: x[1][tname]).map(hpd_map_helper).reduce(hpd_reduce_helper)[1]

    def calc_quantile_multchain(self, quantiles):
        '''
        Helper function to calculate quantiles for multi-chain statistics
        '''
        tname = self.name

        def quantile_map_helper(trace):
            x = trace.copy()
            if x.ndim > 1:
                sx = np.sort(x.T).T
            else:
                sx = np.sort(x)
            try:
                quants = [sx[int(len(sx) * q / 100.0)] for q in quantiles]
                return (1, dict(zip(quantiles, quants)))
            except IndexError:
                return 'Too few elements for quantile calculation'

        def quantile_reduce_helper(x, y):
            average_quantile = dict()
            for k in x[1].keys():
                average_quantile[k] = np.divide(np.add(
                    np.multiply(x[1][k], x[0]), np.multiply(y[1][k], y[0])), x[0] + y[0], dtype=float)
            return (x[0] + y[0], average_quantile)

        return self.db.rdd.map(lambda x: x[1][tname]).map(quantile_map_helper).reduce(quantile_reduce_helper)[1]

    def calc_mean_std(self):
        '''
        Helper function to calculate mean and standard deviation for multi-chain statistics
        '''
        tname = self.name

        def mapper_helper(trace):
            return trace.shape[0], trace.mean(0), trace.std(0)

        def reduce_helper(x, y):
            weighted_mean = np.divide(
                np.multiply(x[1], x[0]) + np.multiply(y[1], y[0]), x[0] + y[0], dtype=float)
            weighted_std = np.add(
                np.multiply(np.square(x[2]), x[0] - 1), np.multiply(np.square(y[2]), y[0] - 1))
            weighted_std = np.add(
                weighted_std, np.multiply(np.square(x[1]), x[0]))
            weighted_std = np.add(
                weighted_std, np.multiply(np.square(y[1]), y[0]))
            weighted_std = np.subtract(
                weighted_std, np.multiply(np.square(weighted_mean), x[0] + y[0]))
            weighted_std = np.divide(
                weighted_std, x[0] + y[0] - 1, dtype=float)
            return (x[0] + y[0], weighted_mean, np.sqrt(weighted_std))
        return self.db.rdd.map(lambda x: x[1][tname]).map(mapper_helper).reduce(reduce_helper)[1:]

    def multiplechain_batchsd(self, batches=5):
        '''
        Helper function to calculate mc error for multi-chain statistics
        '''
        tname = self.name

        def batchsd_helper(trace):
            if batches > len(trace):
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
                    batched_traces = np.resize(
                        trace, (batches, len(trace) / batches))
                except ValueError:
                    # If batches do not divide evenly, trim excess samples
                    resid = len(trace) % batches
                    batched_traces = np.resize(
                        trace[:-resid],
                        (batches,
                         len(trace) / batches))
            means = np.mean(batched_traces, 1)
            return (np.std(means), np.mean(means), len(trace) / batches)

        def reduce_helper(x, y):
            weighted_mean = np.divide(
                np.multiply(x[1], x[2]) + np.multiply(y[1], y[2]), x[2] + y[2], dtype=float)
            weighted_std = np.add(
                np.multiply(np.square(x[0]), x[2] - 1), np.multiply(np.square(y[0]), y[2] - 1))
            weighted_std = np.add(
                weighted_std, np.multiply(np.square(x[1]), x[2]))
            weighted_std = np.add(
                weighted_std, np.multiply(np.square(y[1]), y[2]))
            weighted_std = np.subtract(
                weighted_std, np.multiply(np.square(weighted_mean), x[2] + y[2]))
            weighted_std = np.divide(
                weighted_std, x[2] + y[2] - 1, dtype=float)
            return (weighted_std, weighted_mean, x[2] + y[2])

        return self.db.rdd.map(lambda x: x[1][tname]).map(batchsd_helper).reduce(reduce_helper)[0]

    def batchsd(self, chain=-1, batches=5, rdd=None):
        '''
        Calculates the simulation standard error
        '''
        if rdd is None:
            tname = self.name
            rdd = self.db.rdd.filter(lambda x: x[0] == chain).map(
                lambda x: x[1][tname]).cache()

        def batchsd_helper(trace):
            if batches > len(trace):
                return 'Could not generate mc error'
            if len(np.shape(trace)) > 1:

                dims = np.shape(trace)
                # ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
                ttrace = np.transpose([t.ravel() for t in trace])

                return np.reshape([batchsd_helper(t) for t in ttrace], dims[1:])

            else:
                if batches == 1:
                    return np.std(trace) / np.sqrt(len(trace))

                try:
                    batched_traces = np.resize(
                        trace, (batches, len(trace) / batches))
                except ValueError:
                    # If batches do not divide evenly, trim excess samples
                    resid = len(trace) % batches
                    batched_traces = np.resize(
                        trace[:-resid],
                        (batches,
                         len(trace) / batches))

            means = np.mean(batched_traces, 1)
            return np.std(means) / np.sqrt(batches)
        return rdd.map(batchsd_helper).first()


class Database():

    '''
    Spark Database
    '''

    def __init__(self, rdd, funs_to_tally):
        '''
        Create a database instance (spark)
        '''
        self.__Trace__ = Trace
        self.__name__ = 'spark'
        self.trace_names = funs_to_tally
        self.rdd = rdd
        self._traces = {}
        self.chains = self.rdd.count()
        for tname in self.trace_names:
            if tname not in self._traces:
                self._traces[tname] = self.__Trace__(
                    name=tname, db=self, chain=self.chains)

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
        Return a dictionary containing the state of the Model and its StepMethods
        '''
        last = self.chains - 1
        return self.rdd.filter(lambda x: x[0] == last).map(lambda x: x[1]['_state_']).first()

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
        trace_names = self.trace_names

        def truncate_helper(x):
            if x[0] == chain:
                for tname in trace_names:
                    x[1][tname] = x[1][tname][:index]
            return x
        new_rdd = self.rdd.map(truncate_helper).cache()
        self.rdd = new_rdd


def load_pickle(spark_context, dbname):
    '''
    Create a Database instance from the data stored in a directory on HDFS, where
    data has been stored as a Pickle object

    Parameters
    ----------
    spark_context : SparkContext
            A SparkContext instance that will be used to load data
    dbname : str
            Location of the Pickle object on HDFS
    '''
    data_list = spark_context.pickleFile(name=dbname).collect()
    rdd = spark_context.parallelize(data_list).cache()
    vars_to_tally = rdd.map(lambda x: x[1].keys()).first()
    vars_to_tally.remove('_state_')
    return Database(rdd, vars_to_tally)


def load_txt(spark_context, dbname):
    '''
    Create a Database instance from the data stored in a directory on HDFS, where
    data has been stored in txt files

    Parameters
    ----------
    spark_context : SparkContext
            A SparkContext instance that will be used to load data
    dbname : str
            Location of the Pickle object on HDFS
    '''
    def load_mapper(x):
        dirname = os.path.split(os.path.split(str(x[0]))[0])[1]
        if dirname == 'state':
            from numpy import array
            x = str(x[1])
            data_stream = x.split('\n')
            data_list = list()
            for line in data_stream[:-1]:
                data = eval(line)
                data_list.append((data[0], ('_state_', data[1])))
            return data_list
        else:
            x = str(x[1])
            data_stream = x.split('\n')
            line = 0
            data_list = list()
            while line < len(data_stream) - 1:
                from StringIO import StringIO
                var_name = str(data_stream[line][12:])
                line += 1
                chain = eval(data_stream[line][9:])
                line += 1
                shape = eval(data_stream[line][16:])
                line += 1
                length = shape[0]  # reduce(lambda x,y: x*y, shape)
                line += 1
                data = '\n'.join(data_stream[line:line + length])
                data = np.loadtxt(StringIO(data), delimiter=',').reshape(shape)
                line += length + 1
                data_list.append((chain, (var_name, data)))
            return data_list

    def load_reducer(x, y):
        if isinstance(x, dict):
            if isinstance(y, dict):
                x.update(y)
                return x
            else:
                x[y[0]] = y[1]
            return x
        elif isinstance(y, dict):
            y[x[0]] = x[1]
            return y
        else:
            db = dict()
            db[x[0]] = x[1]
            db[y[0]] = y[1]
            return db
    files = spark_context.wholeTextFiles(os.path.join(dbname, '*'))
    return files.flatMap(load_mapper).reduceByKey(load_reducer).cache()
