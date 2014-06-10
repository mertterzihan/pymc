from pymc.MCMCSpark import MCMCSpark

if __name__ == "__main__":

	db = 'hdfs'
	dbname = 'user/test'

	# Location of the file that defines model
	model_file = 'pymc/examples/disaster_model.py'

	# HDFS Config
	hdfs_host = 'localhost'
	port = '50070'
	user_name = 'test'

	# Spark Config
	master = 'local[4]'
	spark_home = 'spark'

	# Number of times the MCMC will be run on Spark
	nJobs=10

	m = MCMCSpark(db=db, dbname=dbname, model_file=model_file, spark_home=spark_home, spark_host=master, nJobs=nJobs, hdfs_host=hdfs_host, port=port, user_name=user_name)
	db = m.sample(10)
	print db.trace('early_mean')[:]
	print db.trace('early_mean', chain=0)[:]