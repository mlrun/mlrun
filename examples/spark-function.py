# Pyspark example called by mlrun_spark_k8s.ipynb


from mlrun import get_or_create_ctx


# Acquire MLRun context
mlctx = get_or_create_ctx('spark-function')

#Get MLRun parameters
mlctx.logger.info('!@!@!@!@!@ Getting env variables')
READ_OPTIONS = mlctx.get_param('data_sources')
QUERY = mlctx.get_param('query')
WRITE_OPTIONS = mlctx.get_param('write_options')

from pyspark.sql import SparkSession

#Create spark session
spark = SparkSession.builder \
    .appName('Spark function') \
    .getOrCreate()



#Loading data from a JDBC source
for data_source in READ_OPTIONS:
    spark.read.load(**READ_OPTIONS[data_source]).createOrReplaceTempView(data_source)

#Transform the data using SQL query
spark.sql(QUERY).write.save(**WRITE_OPTIONS)

#write the result datadrame to destination
mlctx.logger.info('!@!@!@!@!@ Saved')
spark.stop()
