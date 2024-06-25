# Copyright 2024 Iguazio
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

import mlrun


def get_snowflake_password():
    key = "SNOWFLAKE_PASSWORD"
    snowflake_password = mlrun.get_secret_or_env(key)

    if not snowflake_password:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"No password provided. Set password using the {key} "
            "project secret or environment variable."
        )

    return snowflake_password


def get_snowflake_spark_options(attributes):
    if not attributes:
        return {}
    return {
        "format": "net.snowflake.spark.snowflake",
        "sfURL": attributes.get("url"),
        "sfUser": attributes.get("user"),
        "sfPassword": get_snowflake_password(),
        "sfDatabase": attributes.get("database"),
        "sfSchema": attributes.get("db_schema"),
        "sfWarehouse": attributes.get("warehouse"),
        "application": "iguazio_platform",
        "TIMESTAMP_TYPE_MAPPING": "TIMESTAMP_LTZ",
    }
