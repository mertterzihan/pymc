import pymc as pm
from pymc.examples import disaster_model

# Where to save
dbname = 'user/test3'

# HDFS Config
host = 'localhost'
port = '50070'
user_name = 'test'

# Create and save data to HDFS
M = pm.MCMC(disaster_model, db='hdfs', dbname=dbname, host=host, port=port, user_name=user_name)
M.sample(10)
M.db.close()

# Load data from HDFS
db = pm.database.hdfs.load(dirname=dbname, host=host, port=port, user_name=user_name)
print len(db.trace('rate')[:])