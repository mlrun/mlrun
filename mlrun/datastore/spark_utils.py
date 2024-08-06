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


from typing import Union

import mlrun
from mlrun.features import Entity


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


def check_special_columns_exists(
    spark_df, entities: list[Union[Entity, str]], timestamp_key: str, label_column: str
):
    columns = spark_df.columns
    entities = entities or []
    entities = [
        entity.name if isinstance(entity, Entity) else entity for entity in entities
    ]
    missing_entities = [entity for entity in entities if entity not in columns]
    cases_message = "Please check the letter cases (uppercase or lowercase)"
    if missing_entities:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"There are missing entities from dataframe during ingestion. missing_entities: {missing_entities}."
            f" {cases_message}"
        )
    if timestamp_key and timestamp_key not in columns:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"timestamp_key is missing from dataframe during ingestion. timestamp_key: {timestamp_key}."
            f" {cases_message}"
        )
    if label_column and label_column not in columns:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"label_column is missing from dataframe during ingestion. label_column: {label_column}. "
            f"{cases_message}"
        )
