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
#
# from time import sleep
#
# from pyspark.sql import SparkSession
#
# from mlrun import get_or_create_ctx
#
# context = get_or_create_ctx("spark-function")
#
# # build spark session
# spark = SparkSession.builder.appName("Spark job").getOrCreate()
#
# # log final report
# context.log_result("spark_result", 1000)
# sleep(50)
# spark.stop()


from pyspark.sql import SparkSession


def handler(context, event):
    # # Initialize Spark session
    # spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    #
    # # Example of Spark job: create a DataFrame and show its contents
    # data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    # df = spark.createDataFrame(data, ["name", "value"])
    # df.show()
    #
    # # Stop the Spark session
    # spark.stop()
    spark = SparkSession.builder.appName("saarc").getOrCreate()

    df1 = spark.sparkContext.parallelize([[1, 2, 3], [2, 3, 4]]).toDF(("key", "b", "c"))

    df1.write.format("io.iguaz.v3io.spark.sql.kv").save(
        "v3io://users/admin/test_saar1", mode="overwrite"
    )

    spark.read.format("io.iguaz.v3io.spark.sql.kv").load(
        "v3io://users/admin/test_saar1"
    ).show()

    sleep(50)
    spark.stop()
