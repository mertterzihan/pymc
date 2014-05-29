if __name__ == '__main__': 

    import pymc as pm
    import pymc.examples.hierarchical as hier

    with hier.model:
    	hdfs = pm.backends.HDFS(name='user/test', host='localhost', port='50070', user_name='test')
        trace = pm.sample(100, hier.step, hier.start, trace=hdfs)

        trace2 = pm.backends.hdfs.load(name='user/test', host='localhost', port='50070', user_name='test')
        print trace2.get_values(varname='s')