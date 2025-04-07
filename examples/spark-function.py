# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pyspark example called by mlrun_spark_k8s.ipynb


from pyspark.sql import SparkSession

from mlrun import get_or_create_ctx

# Acquire MLRun context
mlctx = get_or_create_ctx("spark-function")

# Get MLRun parameters
mlctx.logger.info("!@!@!@!@!@ Getting env variables")
READ_OPTIONS = mlctx.get_param("data_sources")
QUERY = mlctx.get_param("query")
WRITE_OPTIONS = mlctx.get_param("write_options")

# Create spark session
spark = SparkSession.builder.appName("Spark function").getOrCreate()


# Loading data from a JDBC source
for data_source in READ_OPTIONS:
    spark.read.load(**READ_OPTIONS[data_source]).createOrReplaceTempView(data_source)

# Transform the data using SQL query
spark.sql(QUERY).write.save(**WRITE_OPTIONS)

# write the result datadrame to destination
mlctx.logger.info("!@!@!@!@!@ Saved")
spark.stop()
