'''
Python module for distributing an MCMC model among Spark clusters. 

It distributes the observation data using Spark's built-in textFile function.
'''

__all__ = ['DistributedMCMC']

from .MCMCSpark import MCMCSpark
from .MCMC import MCMC
from .database import ram, distributed_spark

class DistributedMCMC(MCMCSpark):

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
            - spark_context : SparkContext
                A SparkContext instance that will be used to load data
            - dbname : str
                Optional, location to save the files on HDFS
            - model_function : function
                A wrapper function which builds the model and returns it
            - observation_file : HDFS path
                Path of the data file which will be partitioned over the machines in the cluster
            - local_iter : int
                Number of iterations that local sampler will be run
            - global_update : function
                A wrapper function which takes rdd as a parameter and updates the global parameters
            - step_function : function
                A function that takes an mcmc instance as a parameter and returns an instance with updated step methods
            - data_process : function
                A function which enables the user to preprocess the data, instead of processing it in each iteration
            - sample_return : function
                A function that appends custom data to the returned object from the main mapper that performs sampling
            - save_traces : function
                It enables user to dump the traces to HDFS after each global iteration instead of storing them in memory
        '''
        self.model_function = kwargs.pop("model_function", None)
        self.observation_file = kwargs.pop("observation_file", None)
        self.local_iter = kwargs.pop("local_iter", None)
        self.global_update = kwargs.pop("global_update", None)
        self.step_function = kwargs.pop("step_function", None)
        self.data_process = kwargs.pop("data_process", None)
        self.sample_return = kwargs.pop("sample_return", None)
        self.save_traces = kwargs.pop("save_traces", None)
        MCMCSpark.__init__(self, input=None, db=db, name=name, calc_deviance=calc_deviance, nJobs=nJobs, **kwargs)

    def sample(
        self, iter, burn=0, thin=1, tune_interval=1000, tune_throughout=True,
        save_interval=None, burn_till_tuned=False, stop_tuning_after=5,
            verbose=0, progress_bar=True):
        '''
        Partitions the data over the machines, runs sampling algorithm on the slaves and 
        updates global parameter in turn until reaching the specified number of iterations

        Parameters are similar to MCMC.sample()
        '''
        name = self.name
        calc_deviance = self.calc_deviance
        kwargs = self.kwargs
        model_function = self.model_function
        global_update = self.global_update
        observation_file = self.observation_file
        local_iter = self.local_iter
        nJobs = self.nJobs
        self.total_iter = iter
        sample_return = self.sample_return
        #total_iter = self.total_iter
        step_function = None
        dump_trace = True
        if self.save_traces is None:
            dump_trace = False
        if self.step_function is not None:
            step_function = self.step_function

        def sample_on_spark(data):
            # Load the database, so that MCMC can continue sampling from where it had concluded the previous local iteration
            def load_ram_database(data_dict):
                db = ram.Database('temp_database')
                trace_names = list()
                for key in data_dict.keys():
                    if key != '_state_':
                        trace_names.append(key)
                        db._traces[key] = ram.Trace(name=key, db=db)
                        setattr(db, key, db._traces[key])
                    else:
                        db._state_ = data_dict[key]
                db.trace_names.append(trace_names)
                return db

            if global_param is None:
                input_model = model_function(data, global_param)
            else:
                input_model = model_function(data, global_param.value)
            if len(data) > 2: # If this method has been run in a previous iteration, so that MCMC should be loaded
                index = len(data[2])
                m = MCMC(input_model, db=load_ram_database(data[2][index-1]), name=name, calc_deviance=calc_deviance, **kwargs)
            else: # If this is the first iteration
                m = MCMC(input_model, db='ram', name=name, calc_deviance=calc_deviance, **kwargs)

            if step_function is not None:
                m = step_function(m)

            m.sample(local_iter, burn, thin, tune_interval, tune_throughout,
                     save_interval, burn_till_tuned, stop_tuning_after,
                     verbose, progress_bar)

            # TODO: Local Update

            # Create or update the dictionary
            if len(data) > 2 and not dump_trace:
                import numpy as np
                container_list = data[2]
                container = {}
                for tname in m.db._traces:
                    container[tname] = m.trace(tname, chain=None)[:]
                container['_state_'] = m.get_state()
                container_list.append(container)
                return_data = (data[0], data[1], container_list)
            else:
                container_list = list()
                container = {}
                for tname in m.db._traces:
                    container[tname] = m.trace(tname)[:]
                container['_state_'] = m.get_state()
                container_list.append(container)
                return_data = (data[0], data[1], container_list)
            if sample_return is not None:
                return_data += sample_return(m)
            return return_data


        def generate_keys(splitIndex, iterator):
            for i in iterator:
                yield (splitIndex,i)

        # Partition the data and generate a list of data assigned to each node
        rdd = self.sc.textFile(observation_file, minPartitions=nJobs).mapPartitionsWithIndex(generate_keys).groupByKey()
        if self.data_process is not None:
            data_process = self.data_process
            rdd = rdd.map(data_process)
        current_iter = 0
        while current_iter < self.total_iter:
            # If the user has provided a global update function, execute it to synch the nodes
            if current_iter == 0 or self.global_update is None:
                global_param = None
            else:
                param = self.global_update(rdd)
                global_param = self.sc.broadcast(param) # Broadcast the global parameters
            if self.save_traces is not None:
                old_rdd = rdd
                rdd = rdd.map(sample_on_spark).cache() # Run the local sampler
                self.save_traces(rdd, current_iter, self.local_iter)
                old_rdd.unpersist()
                def mapper(x):
                    d = dict()
                    for key in x[2][0].keys():
                        if key == '_state_':
                            d[key] = x[2][0][key]
                        else:
                            d[key] = None
                    if len(x) == 3:
                        return (x[0], x[1], d)
                    else:
                        return (x[0], x[1], d, x[3])
                #rdd = rdd.map(mapper)
            else:
                rdd = rdd.map(sample_on_spark).cache()
            current_iter += self.local_iter
        rdd = rdd.map(lambda x: (x[0], x[2])).cache()
        def extract_var_names(a,b):
            if isinstance(a, set):
                if isinstance(b, set):
                    a.update(b)
                else:
                    a.add(b)
                return a
            elif isinstance(b, set):
                b.add(a)
                return b
            else:
                s = set([a,b])
                return s
        # Extract the variable names
        vars_to_tally = rdd.map(lambda x: x[1][0]).flatMap(lambda x: filter(lambda i: i!='_state_', x.keys())).reduce(extract_var_names)
        self._variables_to_tally = vars_to_tally
        self._assign_database_backend(rdd, vars_to_tally)
        # If hdfs was selected as the database, save the traces as txt files
        if self.save_to_hdfs:
            self.save_as_txt_file(self.dbname)


    def _assign_database_backend(self, db, vars_to_tally):
        '''
        Assign distributed_spark RDD database
        '''
        self.db = distributed_spark.Database(db, vars_to_tally)


    def save_as_txt_file(self, path, chain=None):
        '''
        Save the data to HDFS as txt files

        Parameters
        ----------
        path : str
            Name of the file to save the data
        chain : int 
            The index of the chain to be saved. Defaults to None (all chains)
        '''
        temp_rdd = self.db.rdd
        if chain is not None:
            if chain < 0:
                chain = xrange(self.db.chains)[chain]
            self.save_txt_helper(path, chain)
        else:
            total_chains = temp_rdd.map(lambda x: len(x[1])).first()
            for chain in xrange(total_chains):
                self.save_txt_helper(path, chain)
            

    def save_txt_helper(self, path, chain):
        '''
        Helper function for saving data to HDFS as txt files
        '''
        import datetime
        import os
        import numpy as np
        from numpy.compat import asstr
        for var in self._variables_to_tally:
            def save_mapper(x):
                data = '# Variable: %s\n' % var
                data += '# Partition: %s\n' % x[0]
                data += '# Sample shape: %s\n' % str(x[1].shape)
                data += '# Date: %s\n' % datetime.datetime.now()
                X = x[1].reshape((-1, x[1][0].size))
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

            self.db.rdd.filter(lambda x: var in x[1][chain]).map(lambda x: (x[0], x[1][chain][var])).map(save_mapper).saveAsTextFile(os.path.join(path, str(chain), var))
        self.db.rdd.map(lambda x: (x[0], x[1][chain]['_state_'])).saveAsTextFile(os.path.join(path, str(chain), 'state'))


    def combine_samples(self, variables=None, chain=None, method='semiparametric'):
        import re
        pattern = re.compile(".*adaptive_scale_factor|deviance")
        method_collection = set(['parametric', 'semiparametric'])
        estimator_dict = dict()
        if variables is None:
            variables = [v for v in self._variables_to_tally if not pattern.match(v)]
        else:
            variables = [v for v in variables if (v in self._variables_to_tally) and (not pattern.match(v))]

        if method not in method_collection:
            raise ValueError("Currently, only semiparametric and parametric estimators are supported!")
        for variable in variables:
            if method == 'semiparametric':
                estimator_dict == self.db._traces[variable].estimate_semiparametric(total_iter=self.total_iter, chain=chain)
            elif method == 'parametric':
                estimator_dict[variable] = self.db._traces[variable].estimate_parametric(chain=chain)
        return estimator_dict

                
