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


import mlrun


def spark_session_update_hadoop_options(session, spark_options) -> dict[str, str]:
    hadoop_conf = session.sparkContext._jsc.hadoopConfiguration()
    non_hadoop_spark_options = {}

    for key, value in spark_options.items():
        if key.startswith("spark.hadoop."):
            key = key[len("spark.hadoop.") :]
            original_value = hadoop_conf.get(key, None)
            if original_value and original_value != value:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The 'spark.hadoop.{key}' value is in conflict due to a discrepancy "
                    "with a previously established setting.\n"
                    f"This issue arises if 'spark.hadoop.{key}' has been preset in the Spark session, "
                    "or when using datastore profiles with differing security settings for this particular key."
                )
            hadoop_conf.set(key, value)
        else:
            non_hadoop_spark_options[key] = value
    return non_hadoop_spark_options
