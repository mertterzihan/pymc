from pymc.MCMCSpark import MCMCSpark
from pymc.examples import disaster_model

input = disaster_model
nJobs = 10
m = MCMCSpark(input=input, nJobs=10, spark_context=sc)
m.sample(50)

m.summary(batches=10)
m.db._traces['early_mean'].stats(batches=10, chain=0)